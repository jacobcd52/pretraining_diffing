"""
Implements the standard SAE training scheme.
"""
import torch as t
from typing import Optional

from trainers.trainer import SAETrainer, get_lr_schedule, get_sparsity_warmup_fn, ConstrainedAdam
from config import DEBUG
from dictionary import AutoEncoder
from collections import namedtuple

class StandardTrainer(SAETrainer):
    """
    Standard SAE training scheme following Towards Monosemanticity. Decoder column norms are constrained to 1.
    """
    def __init__(self,
                 steps: int, # total number of steps to train for
                 activation_dim: int,
                 dict_size: int,
                 layer: int,
                 lm_name: str,
                 dict_class=AutoEncoder,
                 lr:float=1e-3,
                 l1_penalty:float=1e-1,
                 warmup_steps:int=1000, # lr warmup period at start of training and after each resample
                 sparsity_warmup_steps:Optional[int]=2000, # sparsity warmup period at start of training
                 decay_start:Optional[int]=None, # decay learning rate after this many steps
                 resample_steps:Optional[int]=None, # how often to resample neurons
                 seed:Optional[int]=None,
                 device=None,
                 wandb_name:Optional[str]='StandardTrainer',
                 submodule_name:Optional[str]=None,
    ):
        super().__init__(seed)

        assert layer is not None and lm_name is not None
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name

        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)

        # initialize dictionary
        self.ae = dict_class(activation_dim, dict_size)

        self.lr = lr
        self.l1_penalty=l1_penalty
        self.warmup_steps = warmup_steps
        self.sparsity_warmup_steps = sparsity_warmup_steps
        self.steps = steps
        self.decay_start = decay_start
        self.wandb_name = wandb_name

        if device is None:
            self.device = 'cuda' if t.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.ae.to(self.device)

        self.resample_steps = resample_steps
        if self.resample_steps is not None:
            # how many steps since each neuron was last activated?
            self.steps_since_active = t.zeros(self.ae.dict_size, dtype=int).to(self.device)
        else:
            self.steps_since_active = None 

        self.optimizer = ConstrainedAdam(self.ae.parameters(), self.ae.decoder.parameters(), lr=lr)

        lr_fn = get_lr_schedule(steps, warmup_steps, decay_start, resample_steps, sparsity_warmup_steps)
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)

        self.sparsity_warmup_fn = get_sparsity_warmup_fn(steps, sparsity_warmup_steps)

    def resample_neurons(self, deads, activations):
        with t.no_grad():
            if deads.sum() == 0: return
            print(f"resampling {deads.sum().item()} neurons")

            # compute loss for each activation
            losses = (activations - self.ae(activations)).norm(dim=-1)

            # sample input to create encoder/decoder weights from
            n_resample = min([deads.sum(), losses.shape[0]])
            indices = t.multinomial(losses, num_samples=n_resample, replacement=False)
            sampled_vecs = activations[indices]

            # get norm of the living neurons
            alive_norm = self.ae.encoder.weight[~deads].norm(dim=-1).mean()

            # resample first n_resample dead neurons
            deads[deads.nonzero()[n_resample:]] = False
            self.ae.encoder.weight[deads] = sampled_vecs * alive_norm * 0.2
            self.ae.decoder.weight[:,deads] = (sampled_vecs / sampled_vecs.norm(dim=-1, keepdim=True)).T
            self.ae.encoder.bias[deads] = 0.


            # reset Adam parameters for dead neurons
            state_dict = self.optimizer.state_dict()['state']
            ## encoder weight
            state_dict[1]['exp_avg'][deads] = 0.
            state_dict[1]['exp_avg_sq'][deads] = 0.
            ## encoder bias
            state_dict[2]['exp_avg'][deads] = 0.
            state_dict[2]['exp_avg_sq'][deads] = 0.
            ## decoder weight
            state_dict[3]['exp_avg'][:,deads] = 0.
            state_dict[3]['exp_avg_sq'][:,deads] = 0.
    
    def loss(self, x, step: int, logging=False, **kwargs):

        sparsity_scale = self.sparsity_warmup_fn(step)

        x_hat, f = self.ae(x, output_features=True)
        l2_loss = t.linalg.norm(x - x_hat, dim=-1).mean()
        recon_loss = (x - x_hat).pow(2).sum(dim=-1).mean()
        l1_loss = f.norm(p=1, dim=-1).mean()

        if self.steps_since_active is not None:
            # update steps_since_active
            deads = (f == 0).all(dim=0)
            self.steps_since_active[deads] += 1
            self.steps_since_active[~deads] = 0
        
        loss = recon_loss + self.l1_penalty * sparsity_scale * l1_loss

        if not logging:
            return loss
        else:
            return namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])(
                x, x_hat, f,
                {
                    'l2_loss' : l2_loss.item(),
                    'mse_loss' : recon_loss.item(),
                    'sparsity_loss' : l1_loss.item(),
                    'loss' : loss.item()
                }
            )


    def update(self, step, activations):
        activations = activations.to(self.device)

        self.optimizer.zero_grad()
        loss = self.loss(activations, step=step)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        if self.resample_steps is not None and step % self.resample_steps == 0:
            self.resample_neurons(self.steps_since_active > self.resample_steps / 2, activations)

    @property
    def config(self):
        return {
            'dict_class': 'AutoEncoder',
            'trainer_class' : 'StandardTrainer',
            'activation_dim': self.ae.activation_dim,
            'dict_size': self.ae.dict_size,
            'lr' : self.lr,
            'l1_penalty' : self.l1_penalty,
            'warmup_steps' : self.warmup_steps,
            'resample_steps' : self.resample_steps,
            'sparsity_warmup_steps' : self.sparsity_warmup_steps,
            'steps' : self.steps,
            'decay_start' : self.decay_start,
            'seed' : self.seed,
            'device' : self.device,
            'layer' : self.layer,
            'lm_name' : self.lm_name,
            'wandb_name': self.wandb_name,
            'submodule_name': self.submodule_name,
        }


class StandardTrainerAprilUpdate(SAETrainer):
    """
    Standard SAE training scheme following the Anthropic April update. Decoder column norms are NOT constrained to 1.
    This trainer does not support resampling or ghost gradients. This trainer will have fewer dead neurons than the standard trainer.
    """
    def __init__(self,
                steps: int, # total number of steps to train for
                activation_dim: int,
                dict_size: int,
                layer: int,
                lm_name: str,
                dict_class=AutoEncoder,
                lr:float=1e-3,
                l1_penalty:float=1e-1,
                warmup_steps:int=1000, # lr warmup period at start of training
                sparsity_warmup_steps:Optional[int]=2000, # sparsity warmup period at start of training
                decay_start:Optional[int]=None, # decay learning rate after this many steps
                seed:Optional[int]=None,
                device=None,
                wandb_name:Optional[str]='StandardTrainerAprilUpdate',
                submodule_name:Optional[str]=None,
                frac_features_shared:float=0, # fraction of features to be shared across models
                shared_l1_penalty:Optional[float]=None, # l1 penalty for shared features
    ):
        super().__init__(seed)

        assert layer is not None and lm_name is not None
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        
        # Check that shared features parameters are valid
        self.frac_features_shared = frac_features_shared
        self.shared_l1_penalty = shared_l1_penalty
        if frac_features_shared > 0:
            assert dict_class == AutoEncoder, "dict_class must be AutoEncoder when frac_features_shared > 0"
            assert shared_l1_penalty is not None, "shared_l1_penalty must be set when frac_features_shared > 0"
            # Number of features that will be shared
            self.num_shared_features = int(dict_size * frac_features_shared)
            assert self.num_shared_features > 0, "At least one feature must be shared when frac_features_shared > 0"

        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)

        # initialize dictionary
        self.ae = dict_class(activation_dim, dict_size)

        self.lr = lr
        self.l1_penalty=l1_penalty
        self.warmup_steps = warmup_steps
        self.sparsity_warmup_steps = sparsity_warmup_steps
        self.steps = steps
        self.decay_start = decay_start
        self.wandb_name = wandb_name

        if device is None:
            self.device = 'cuda' if t.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.ae.to(self.device)

        self.optimizer = t.optim.Adam(self.ae.parameters(), lr=lr)

        lr_fn = get_lr_schedule(steps, warmup_steps, decay_start, resample_steps=None, sparsity_warmup_steps=sparsity_warmup_steps)
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)

        self.sparsity_warmup_fn = get_sparsity_warmup_fn(steps, sparsity_warmup_steps)

    def loss(self, x, step: int, logging=False, **kwargs):
        sparsity_scale = self.sparsity_warmup_fn(step)

        x_hat, f = self.ae(x, output_features=True)
        l2_loss = t.linalg.norm(x - x_hat, dim=-1).mean()
        recon_loss = (x - x_hat).pow(2).sum(dim=-1).mean()
        
        # Get the number of models and dimensions
        activation_dim = self.ae.activation_dim
        feature_dim = self.ae.dict_size
        n_models = x.shape[1] // activation_dim
        
        # Calculate L1 loss with separate decoder norms for each model's subspace
        if self.frac_features_shared > 0:
            # Initialize separate L1 losses for shared and non-shared features
            shared_l1_loss = 0
            non_shared_l1_loss = 0
            
            # For each model subspace
            for k in range(n_models):
                # Get the model's subspace in the decoder weights
                start_idx = k * activation_dim
                end_idx = (k + 1) * activation_dim
                model_decoder_weights = self.ae.decoder.weight[start_idx:end_idx, :]
                
                # Calculate norms for this model's subspace
                model_weight_norms = model_decoder_weights.norm(p=2, dim=0)
                
                # Calculate L1 loss for shared features
                shared_l1_loss += (f[:, :self.num_shared_features] * model_weight_norms[:self.num_shared_features]).sum(dim=-1).mean()
                
                # Calculate L1 loss for non-shared features
                non_shared_l1_loss += (f[:, self.num_shared_features:] * model_weight_norms[self.num_shared_features:]).sum(dim=-1).mean()
            
            # Total L1 loss (for logging)
            l1_loss = shared_l1_loss + non_shared_l1_loss
            
            # Total loss with different penalties
            loss = (recon_loss + 
                    self.shared_l1_penalty * sparsity_scale * shared_l1_loss + 
                    self.l1_penalty * sparsity_scale * non_shared_l1_loss)
        else:
            # Standard implementation with summed norms across model subspaces
            l1_loss = 0
            
            # For each model subspace
            for k in range(n_models):
                # Get the model's subspace in the decoder weights
                start_idx = k * activation_dim
                end_idx = (k + 1) * activation_dim
                model_decoder_weights = self.ae.decoder.weight[start_idx:end_idx]
                
                # Add L1 loss from this model's subspace
                l1_loss += (f * model_decoder_weights.norm(p=2, dim=0)).sum(dim=-1).mean()
            
            loss = recon_loss + self.l1_penalty * sparsity_scale * l1_loss

        if not logging:
            return loss
        else:
            return namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])(
                x, x_hat, f,
                {
                    'l2_loss' : l2_loss.item(),
                    'mse_loss' : recon_loss.item(),
                    'sparsity_loss' : l1_loss.item(),
                    'loss' : loss.item()
                }
            )


    def update(self, step, activations):
        activations = activations.to(self.device)

        self.optimizer.zero_grad()
        loss = self.loss(activations, step=step)
        loss.backward()
        t.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
        self.optimizer.step()
        
        # Enforce weight sharing for shared features after each optimization step
        if self.frac_features_shared > 0:
            with t.no_grad():
                activation_dim = self.ae.activation_dim
                n_models = activation_dim // self.ae.decoder.weight.shape[0]
                d_model = activation_dim // n_models
                
                # For each shared feature, average its weights across all model subspaces
                for i in range(self.num_shared_features):
                    # Get all weights for this feature across different model subspaces
                    feature_weights = []
                    for k in range(n_models):
                        feature_weights.append(self.ae.decoder.weight[:, i + k * d_model])
                    
                    # Calculate the average weight
                    avg_weight = t.stack(feature_weights).mean(dim=0)
                    
                    # Set all weights for this feature to the average
                    for k in range(n_models):
                        self.ae.decoder.weight[:, i + k * d_model] = avg_weight
        
        self.scheduler.step()

    @property
    def config(self):
        return {
            'dict_class': 'AutoEncoder',
            'trainer_class' : 'StandardTrainerAprilUpdate',
            'activation_dim': self.ae.activation_dim,
            'dict_size': self.ae.dict_size,
            'lr' : self.lr,
            'l1_penalty' : self.l1_penalty,
            'warmup_steps' : self.warmup_steps,
            'sparsity_warmup_steps' : self.sparsity_warmup_steps,
            'steps' : self.steps,
            'decay_start' : self.decay_start,
            'seed' : self.seed,
            'device' : self.device,
            'layer' : self.layer,
            'lm_name' : self.lm_name,
            'wandb_name': self.wandb_name,
            'submodule_name': self.submodule_name,
            'frac_features_shared': self.frac_features_shared,
            'shared_l1_penalty': self.shared_l1_penalty,
            'num_shared_features': getattr(self, 'num_shared_features', 0),
        }

