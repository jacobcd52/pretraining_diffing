"""
Implements the SAE training scheme from https://arxiv.org/abs/2406.04093.
Significant portions of this code have been copied from https://github.com/EleutherAI/sae/blob/main/sae
"""

import einops
import torch as t
import torch.nn as nn
from collections import namedtuple
from typing import Optional

from config import DEBUG
from dictionary import Dictionary
from trainers.trainer import (
    SAETrainer,
    get_lr_schedule,
    set_decoder_norm_to_unit_norm,
    remove_gradient_parallel_to_decoder_directions,
)


@t.no_grad()
def geometric_median(points: t.Tensor, max_iter: int = 100, tol: float = 1e-5):
    """Compute the geometric median `points`. Used for initializing decoder bias."""
    # Initialize our guess as the mean of the points
    guess = points.mean(dim=0)
    prev = t.zeros_like(guess)

    # Weights for iteratively reweighted least squares
    weights = t.ones(len(points), device=points.device)

    for _ in range(max_iter):
        prev = guess

        # Compute the weights
        weights = 1 / t.norm(points - guess, dim=1)

        # Normalize the weights
        weights /= weights.sum()

        # Compute the new geometric median
        guess = (weights.unsqueeze(1) * points).sum(dim=0)

        # Early stopping condition
        if t.norm(guess - prev) < tol:
            break

    return guess


class AutoEncoderTopK(Dictionary, nn.Module):
    """
    The top-k autoencoder architecture and initialization used in https://arxiv.org/abs/2406.04093
    """

    def __init__(self, activation_dim: int, dict_size: int, k: int):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size

        assert isinstance(k, int) and k > 0, f"k={k} must be a positive integer"
        self.register_buffer("k", t.tensor(k, dtype=t.int))
        self.register_buffer("threshold", t.tensor(-1.0, dtype=t.float32))
        
        # Initialize buffers but don't set their shape yet - will be set when loading state dict
        self.register_buffer("act_mean", None)
        self.register_buffer("act_cov_inv_sqrt", None)

        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        self.decoder.weight.data = set_decoder_norm_to_unit_norm(
            self.decoder.weight, activation_dim, dict_size
        )

        self.encoder = nn.Linear(activation_dim, dict_size)
        self.encoder.weight.data = self.decoder.weight.T.clone()
        self.encoder.bias.data.zero_()

        self.b_dec = nn.Parameter(t.zeros(activation_dim))

    def _resize_buffers(self, state_dict):
        """Helper method to resize buffers based on loaded state dict"""
        if 'act_mean' in state_dict and 'act_cov_inv_sqrt' in state_dict:
            mean_shape = state_dict['act_mean'].shape
            cov_shape = state_dict['act_cov_inv_sqrt'].shape
            self.register_buffer("act_mean", t.zeros(mean_shape))
            self.register_buffer("act_cov_inv_sqrt", t.zeros(cov_shape))
        else:
            # If the buffers are not in the state dict, set them to None
            self.register_buffer("act_mean", None)
            self.register_buffer("act_cov_inv_sqrt", None)

    def load_state_dict(self, state_dict, strict: bool = True):
        """Override load_state_dict to handle buffer resizing"""
        self._resize_buffers(state_dict)
        return super().load_state_dict(state_dict, strict)

    def encode(self, x: t.Tensor, return_topk: bool = False, use_threshold: bool = False):
        post_relu_feat_acts_BF = nn.functional.relu(self.encoder(x - self.b_dec))

        if use_threshold:
            encoded_acts_BF = post_relu_feat_acts_BF * (post_relu_feat_acts_BF > self.threshold)
            if return_topk:
                post_topk = post_relu_feat_acts_BF.topk(self.k, sorted=False, dim=-1)
                return encoded_acts_BF, post_topk.values, post_topk.indices, post_relu_feat_acts_BF
            else:
                return encoded_acts_BF

        post_topk = post_relu_feat_acts_BF.topk(self.k, sorted=False, dim=-1)

        # We can't split immediately due to nnsight
        tops_acts_BK = post_topk.values
        top_indices_BK = post_topk.indices

        buffer_BF = t.zeros_like(post_relu_feat_acts_BF)
        encoded_acts_BF = buffer_BF.scatter_(dim=-1, index=top_indices_BK, src=tops_acts_BK)

        if return_topk:
            return encoded_acts_BF, tops_acts_BK, top_indices_BK, post_relu_feat_acts_BF
        else:
            return encoded_acts_BF

    def decode(self, x: t.Tensor) -> t.Tensor:
        return self.decoder(x) + self.b_dec

    def forward(self, x: t.Tensor, output_features: bool = False):
        encoded_acts_BF = self.encode(x)
        x_hat_BD = self.decode(encoded_acts_BF)
        if not output_features:
            return x_hat_BD
        else:
            return x_hat_BD, encoded_acts_BF

    def scale_biases(self, scale: float):
        self.encoder.bias.data *= scale
        self.b_dec.data *= scale
        if self.threshold >= 0:
            self.threshold *= scale

    @classmethod
    def from_pretrained(cls, path, k: Optional[int] = None, device=None):
        """
        Load a pretrained autoencoder from a file.
        """
        state_dict = t.load(path)
        dict_size, activation_dim = state_dict["encoder.weight"].shape

        if k is None:
            k = state_dict["k"].item()
        elif "k" in state_dict and k != state_dict["k"].item():
            raise ValueError(f"k={k} != {state_dict['k'].item()}=state_dict['k']")

        autoencoder = cls(activation_dim, dict_size, k)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder

    @classmethod
    def from_hf(cls, repo_id: str, trainer_idx: int = 0, k: Optional[int] = None, device=None,
    weights_filename: Optional[str] = None, config_filename: Optional[str] = None, 
    is_eleuther: bool = False, threshold: float = -1.0):
        """
        Load a pretrained autoencoder from HuggingFace Hub.
        
        Args:
            repo_id: str, the HuggingFace repository ID (e.g., "username/repo-name")
            trainer_idx: int, which trainer's model to load if multiple were saved
            k: Optional[int], override the k value from the saved model
            device: Optional[str], device to load the model to
            weights_filename: Optional[str], custom filename for the weights file
            config_filename: Optional[str], custom filename for the config file
            is_eleuther: bool, whether the weights are in Eleuther format
            threshold: float, threshold value to use for Eleuther format weights
        """
        from huggingface_hub import hf_hub_download
        import os
        import json
        
        if weights_filename is None:
            # Try to find the appropriate weights file with either extension
            for extension in [".pt", ".safetensors"]:
                try:
                    weights_filename = f"trainer_{trainer_idx}/ae{extension}"
                    # Attempt to download to see if file exists
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=weights_filename,
                        repo_type="model"
                    )
                    # If we get here, the file exists
                    break
                except Exception:
                    weights_filename = None
                    continue
                    
            # If we still don't have a weights filename, use the default .pt
            if weights_filename is None:
                weights_filename = f"trainer_{trainer_idx}/ae.pt"
                
        if config_filename is None:
            config_filename = f"trainer_{trainer_idx}/config.json"
        
        try:
            # Download the config file first to get k value
            config_path = hf_hub_download(
                repo_id=repo_id,
                filename=config_filename,
                repo_type="model"
            )
            
            # Load config to get k
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Use provided k or the one from config
            config_k = config.get("k")
            if k is None and config_k is not None:
                k = config_k
            
            # Download the model file
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=weights_filename,
                repo_type="model"
            )
            
            # Load the model based on file extension
            file_extension = os.path.splitext(model_path)[1].lower()
            if file_extension == '.safetensors':
                try:
                    import safetensors.torch
                    # Load the safetensors file
                    state_dict = safetensors.torch.load_file(model_path)
                    
                    # Handle Eleuther format if needed
                    if is_eleuther:
                        # Determine dimensions from the state dict
                        if 'encoder.weight' in state_dict:
                            dict_size, activation_dim = state_dict['encoder.weight'].shape
                        elif 'W_dec' in state_dict:
                            activation_dim, dict_size = state_dict['W_dec'].shape
                        else:
                            raise ValueError("Cannot determine model dimensions from state dict")
                        
                        # Create the model
                        autoencoder = cls(activation_dim, dict_size, k if k is not None else 1)
                        
                        # Convert the state dict
                        converted_dict = autoencoder.convert_eleuther(state_dict, k, threshold)
                        
                        # Load the converted state dict
                        autoencoder.load_state_dict(converted_dict)
                    else:
                        # Then create the model from the state dict manually
                        dict_size, activation_dim = state_dict["encoder.weight"].shape
                        
                        autoencoder = cls(activation_dim, dict_size, k)
                        autoencoder.load_state_dict(state_dict)
                    
                    if device is not None:
                        autoencoder.to(device)
                except ImportError:
                    raise ImportError(
                        "safetensors package is required to load .safetensors files. "
                        "Install with: pip install safetensors"
                    )
            else:  # Default to .pt loading
                # Load the state dict
                state_dict = t.load(model_path)
                
                if is_eleuther:
                    # Determine dimensions from the state dict
                    if 'encoder.weight' in state_dict:
                        dict_size, activation_dim = state_dict['encoder.weight'].shape
                    elif 'W_dec' in state_dict:
                        activation_dim, dict_size = state_dict['W_dec'].shape
                    else:
                        raise ValueError("Cannot determine model dimensions from state dict")
                    
                    # Create the model
                    autoencoder = cls(activation_dim, dict_size, k if k is not None else 1)
                    
                    # Convert the state dict
                    converted_dict = autoencoder.convert_eleuther(state_dict, k, threshold)
                    
                    # Load the converted state dict
                    autoencoder.load_state_dict(converted_dict)
                else:
                    autoencoder = cls.from_pretrained(model_path, k=k, device=device)
            
            # Attach config to the model
            autoencoder.config = config
            
            return autoencoder
            
        except Exception as e:
            raise RuntimeError(f"Error loading model from HuggingFace Hub: {str(e)}")
        
    def convert_eleuther(self, state_dict, k=None, threshold=-1.0):
        """
        Convert a state dict from Eleuther format to the format expected by this model.
        
        Args:
            state_dict: dict, the state dictionary in Eleuther format with keys like 
                    'W_dec', 'b_dec', 'encoder.bias', 'encoder.weight'
            k: Optional[int], the value of k to use if not present in the state dict
            threshold: float, the threshold value to use if not present in the state dict
        
        Returns:
            dict: A converted state dictionary compatible with this model
        """
        converted_dict = {}
        
        # Handle W_dec tensor which needs to be transposed
        if 'W_dec' in state_dict:
            # W_dec in Eleuther format is [activation_dim, dict_size]
            # decoder.weight in our format is [dict_size, activation_dim]
            converted_dict['decoder.weight'] = state_dict['W_dec'].T
        
        # Handle other tensors that can be copied directly
        for old_key, new_key in {
            'b_dec': 'b_dec',
            'encoder.bias': 'encoder.bias',
            'encoder.weight': 'encoder.weight'
        }.items():
            if old_key in state_dict:
                converted_dict[new_key] = state_dict[old_key]
        
        # Set k if not in state_dict
        if k is not None:
            converted_dict['k'] = t.tensor(k, dtype=t.int)
        elif 'k' in state_dict:
            converted_dict['k'] = state_dict['k']
        else:
            # Use the current k value if available
            converted_dict['k'] = self.k
        
        # Set threshold if not in state_dict
        converted_dict['threshold'] = t.tensor(threshold, dtype=t.float32)
        
        # If act_mean and act_cov_inv_sqrt are in the original state_dict, copy them
        if 'act_mean' in state_dict:
            converted_dict['act_mean'] = state_dict['act_mean']
        
        if 'act_cov_inv_sqrt' in state_dict:
            converted_dict['act_cov_inv_sqrt'] = state_dict['act_cov_inv_sqrt']
        
        return converted_dict


class TopKTrainer(SAETrainer):
    """
    Top-K SAE training scheme.
    """

    def __init__(
        self,
        steps: int,  # total number of steps to train for
        activation_dim: int,
        dict_size: int,
        k: int,
        layer: int,
        lm_name: str,
        dict_class: type = AutoEncoderTopK,
        lr: Optional[float] = None,
        auxk_alpha: float = 1 / 32,  # see Appendix A.2
        warmup_steps: int = 1000,
        decay_start: Optional[int] = None,  # when does the lr decay start
        threshold_beta: float = 0.999,
        threshold_start_step: int = 1000,
        seed: Optional[int] = None,
        device: Optional[str] = None,
        wandb_name: str = "AutoEncoderTopK",
        submodule_name: Optional[str] = None,
    ):
        super().__init__(seed)

        assert layer is not None and lm_name is not None
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name

        self.wandb_name = wandb_name
        self.steps = steps
        self.decay_start = decay_start
        self.warmup_steps = warmup_steps
        self.k = k
        self.threshold_beta = threshold_beta
        self.threshold_start_step = threshold_start_step

        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)

        # Initialise autoencoder
        self.ae = dict_class(activation_dim, dict_size, k)
        if device is None:
            self.device = "cuda" if t.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.ae.to(self.device)

        if lr is not None:
            self.lr = lr
        else:
            # Auto-select LR using 1 / sqrt(d) scaling law from Figure 3 of the paper
            scale = dict_size / (2**14)
            self.lr = 2e-4 / scale**0.5

        self.auxk_alpha = auxk_alpha
        self.dead_feature_threshold = 10_000_000
        self.top_k_aux = activation_dim // 2  # Heuristic from B.1 of the paper
        self.num_tokens_since_fired = t.zeros(dict_size, dtype=t.long, device=device)
        self.logging_parameters = ["effective_l0", "dead_features", "pre_norm_auxk_loss"]
        self.effective_l0 = -1
        self.dead_features = -1
        self.pre_norm_auxk_loss = -1

        # Optimizer and scheduler
        self.optimizer = t.optim.Adam(self.ae.parameters(), lr=self.lr, betas=(0.9, 0.999))

        lr_fn = get_lr_schedule(steps, warmup_steps, decay_start=decay_start)

        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)

    def get_auxiliary_loss(self, residual_BD: t.Tensor, post_relu_acts_BF: t.Tensor):
        dead_features = self.num_tokens_since_fired >= self.dead_feature_threshold
        self.dead_features = int(dead_features.sum())

        if self.dead_features > 0:
            k_aux = min(self.top_k_aux, self.dead_features)

            auxk_latents = t.where(dead_features[None], post_relu_acts_BF, -t.inf)

            # Top-k dead latents
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)

            auxk_buffer_BF = t.zeros_like(post_relu_acts_BF)
            auxk_acts_BF = auxk_buffer_BF.scatter_(dim=-1, index=auxk_indices, src=auxk_acts)

            # Note: decoder(), not decode(), as we don't want to apply the bias
            x_reconstruct_aux = self.ae.decoder(auxk_acts_BF)
            l2_loss_aux = (
                (residual_BD.float() - x_reconstruct_aux.float()).pow(2).sum(dim=-1).mean()
            )

            self.pre_norm_auxk_loss = l2_loss_aux

            # normalization from OpenAI implementation: https://github.com/openai/sparse_autoencoder/blob/main/sparse_autoencoder/kernels.py#L614
            residual_mu = residual_BD.mean(dim=0)[None, :].broadcast_to(residual_BD.shape)
            loss_denom = (residual_BD.float() - residual_mu.float()).pow(2).sum(dim=-1).mean()
            normalized_auxk_loss = l2_loss_aux / loss_denom

            return normalized_auxk_loss.nan_to_num(0.0)
        else:
            self.pre_norm_auxk_loss = -1
            return t.tensor(0, dtype=residual_BD.dtype, device=residual_BD.device)

    def update_threshold(self, top_acts_BK: t.Tensor):
        device_type = "cuda" if top_acts_BK.is_cuda else "cpu"
        with t.autocast(device_type=device_type, enabled=False), t.no_grad():
            active = top_acts_BK.clone().detach()
            active[active <= 0] = float("inf")
            min_activations = active.min(dim=1).values.to(dtype=t.float32)
            min_activation = min_activations.mean()

            B, K = active.shape
            assert len(active.shape) == 2
            assert min_activations.shape == (B,)

            if self.ae.threshold < 0:
                self.ae.threshold = min_activation
            else:
                self.ae.threshold = (self.threshold_beta * self.ae.threshold) + (
                    (1 - self.threshold_beta) * min_activation
                )

    def loss(self, x, step=None, logging=False):
        # Run the SAE
        f, top_acts_BK, top_indices_BK, post_relu_acts_BF = self.ae.encode(
            x, return_topk=True, use_threshold=False
        )

        if step > self.threshold_start_step:
            self.update_threshold(top_acts_BK)

        x_hat = self.ae.decode(f)

        # Measure goodness of reconstruction
        e = x - x_hat

        # Update the effective L0 (again, should just be K)
        self.effective_l0 = top_acts_BK.size(1)

        # Update "number of tokens since fired" for each features
        num_tokens_in_step = x.size(0)
        did_fire = t.zeros_like(self.num_tokens_since_fired, dtype=t.bool)
        did_fire[top_indices_BK.flatten()] = True
        self.num_tokens_since_fired += num_tokens_in_step
        self.num_tokens_since_fired[did_fire] = 0

        l2_loss = e.pow(2).sum(dim=-1).mean()
        auxk_loss = (
            self.get_auxiliary_loss(e.detach(), post_relu_acts_BF) if self.auxk_alpha > 0 else 0
        )

        loss = l2_loss + self.auxk_alpha * auxk_loss

        if not logging:
            return loss
        else:
            return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(
                x,
                x_hat,
                f,
                {"l2_loss": l2_loss.item(), "auxk_loss": auxk_loss.item(), "loss": loss.item()},
            )

    def update(self, step, x):
        # Initialise the decoder bias
        if step == 0:
            median = geometric_median(x)
            median = median.to(self.ae.b_dec.dtype)
            self.ae.b_dec.data = median

        # compute the loss
        x = x.to(self.device)
        loss = self.loss(x, step=step)
        loss.backward()

        # clip grad norm and remove grads parallel to decoder directions
        self.ae.decoder.weight.grad = remove_gradient_parallel_to_decoder_directions(
            self.ae.decoder.weight,
            self.ae.decoder.weight.grad,
            self.ae.activation_dim,
            self.ae.dict_size,
        )
        t.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)

        # do a training step
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()

        # Make sure the decoder is still unit-norm
        self.ae.decoder.weight.data = set_decoder_norm_to_unit_norm(
            self.ae.decoder.weight, self.ae.activation_dim, self.ae.dict_size
        )

        return loss.item()

    @property
    def config(self):
        return {
            "trainer_class": "TopKTrainer",
            "dict_class": "AutoEncoderTopK",
            "lr": self.lr,
            "steps": self.steps,
            "auxk_alpha": self.auxk_alpha,
            "warmup_steps": self.warmup_steps,
            "decay_start": self.decay_start,
            "threshold_beta": self.threshold_beta,
            "threshold_start_step": self.threshold_start_step,
            "seed": self.seed,
            "activation_dim": self.ae.activation_dim,
            "dict_size": self.ae.dict_size,
            "k": self.ae.k.item(),
            "device": self.device,
            "layer": self.layer,
            "lm_name": self.lm_name,
            "wandb_name": self.wandb_name,
            "submodule_name": self.submodule_name,
        }
