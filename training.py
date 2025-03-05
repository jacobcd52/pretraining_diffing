"""
Training dictionaries
"""

import json
import torch.multiprocessing as mp
import os
from queue import Empty
from typing import Optional
from contextlib import nullcontext
from huggingface_hub import HfApi

import torch as t
from tqdm import tqdm

import wandb

from dictionary import AutoEncoder
from evaluation import evaluate
from trainers.standard import StandardTrainer


def new_wandb_process(config, log_queue, entity, project):
    wandb.init(entity=entity, project=project, config=config, name=config["wandb_name"])
    while True:
        try:
            log = log_queue.get(timeout=1)
            if log == "DONE":
                break
            wandb.log(log)
        except Empty:
            continue
    wandb.finish()


def log_stats(
    trainers,
    step: int,
    act: t.Tensor,
    activations_split_by_head: bool,
    transcoder: bool,
    log_queues: list=[],
    verbose: bool=False,
):
    with t.no_grad():
        # quick hack to make sure all trainers get the same x
        z = act.clone()
        for i, trainer in enumerate(trainers):
            log = {}
            act = z.clone()
            if activations_split_by_head:  # x.shape: [batch, pos, n_heads, d_head]
                act = act[..., i, :]
            if not transcoder:
                act, act_hat, f, losslog = trainer.loss(act, step=step, logging=True)

                # L0
                l0 = (f != 0).float().sum(dim=-1).mean().item()
                # fraction of variance unexplained
                total_variance = t.var(act, dim=0).sum()
                residual_variance = t.var(act - act_hat, dim=0).sum()
                fvu = residual_variance / total_variance
                log["FVU"] = fvu.item()
            else:  # transcoder
                x, x_hat, f, losslog = trainer.loss(act, step=step, logging=True)

                # L0
                l0 = (f != 0).float().sum(dim=-1).mean().item()

            if verbose:
                print(f"Step {step}: L0 = {l0}, fvu = {fvu}")

            # log parameters from training
            log.update({f"{k}": v.cpu().item() if isinstance(v, t.Tensor) else v for k, v in losslog.items()})
            log[f"l0"] = l0
            trainer_log = trainer.get_logging_parameters()
            for name, value in trainer_log.items():
                if isinstance(value, t.Tensor):
                    value = value.cpu().item()
                log[f"{name}"] = value

            if log_queues:
                log_queues[i].put(log)

def get_norm_factor(data, steps: int) -> float:
    """Per Section 3.1, find a fixed scalar factor so activation vectors have unit mean squared norm.
    This is very helpful for hyperparameter transfer between different layers and models.
    Use more steps for more accurate results.
    https://arxiv.org/pdf/2408.05147
    
    If experiencing troubles with hyperparameter transfer between models, it may be worth instead normalizing to the square root of d_model.
    https://transformer-circuits.pub/2024/april-update/index.html#training-saes"""
    total_mean_squared_norm = 0
    count = 0

    for step, act_BD in enumerate(tqdm(data, total=steps, desc="Calculating norm factor")):
        if step > steps:
            break

        count += 1
        mean_squared_norm = t.mean(t.sum(act_BD ** 2, dim=1))
        total_mean_squared_norm += mean_squared_norm

    average_mean_squared_norm = total_mean_squared_norm / count
    norm_factor = t.sqrt(average_mean_squared_norm).item()

    print(f"Average mean squared norm: {average_mean_squared_norm}")
    print(f"Norm factor: {norm_factor}")
    
    return norm_factor



def trainSAE(
    data,
    trainer_configs: list[dict],
    steps: int,
    use_wandb:bool=False,
    wandb_entity:str="",
    wandb_project:str="",
    save_steps:Optional[list[int]]=None,
    save_dir:Optional[str]=None,
    hf_repo_out:Optional[str]=None,  # New parameter
    log_steps:Optional[int]=None,
    activations_split_by_head:bool=False,
    transcoder:bool=False,
    run_cfg:dict={},
    normalize_activations:bool=False,
    verbose:bool=False,
    device:str="cuda",
    autocast_dtype: t.dtype = t.float32,
):
    """
    Train SAEs using the given trainers

    If normalize_activations is True, the activations will be normalized to have unit mean squared norm.
    The autoencoders weights will be scaled before saving, so the activations don't need to be scaled during inference.
    This is very helpful for hyperparameter transfer between different layers and models.

    Setting autocast_dtype to t.bfloat16 provides a significant speedup with minimal change in performance.

    Args:
        ...existing args...
        hf_repo_out: Optional[str], if provided, will upload the trained model(s) to this HF repo
                     Format should be "username/repo-name" or "organization/repo-name"
    """

    device_type = "cuda" if "cuda" in device else "cpu"
    autocast_context = nullcontext() if device_type == "cpu" else t.autocast(device_type=device_type, dtype=autocast_dtype)

    trainers = []
    for i, config in enumerate(trainer_configs):
        if "wandb_name" in config:
            config["wandb_name"] = f"{config['wandb_name']}_trainer_{i}"
        trainer_class = config["trainer"]
        del config["trainer"]
        trainers.append(trainer_class(**config))

    # If the data buffer has rescaling statistics, transfer them to the autoencoders
        if hasattr(data, 'rescale_acts') and data.rescale_acts:
            for trainer in trainers:
                # Transfer the statistics
                trainer.ae.act_mean = data.act_mean.clone()
                trainer.ae.act_cov_inv_sqrt = data.act_cov_inv_sqrt.clone()
                
                # Update config to record that rescaling was used
                trainer.config['used_buffer_rescaling'] = True
                trainer.config['buffer_rescaling_stats'] = {
                    'mean_shape': list(data.act_mean.shape),
                    'cov_shape': list(data.act_cov_inv_sqrt.shape)
                }

    wandb_processes = []
    log_queues = []

    if use_wandb:
        # Note: If encountering wandb and CUDA related errors, try setting start method to spawn in the if __name__ == "__main__" block
        # https://docs.python.org/3/library/multiprocessing.html#multiprocessing.set_start_method
        # Everything should work fine with the default fork method but it may not be as robust
        for i, trainer in enumerate(trainers):
            log_queue = mp.Queue()
            log_queues.append(log_queue)
            wandb_config = trainer.config | run_cfg
            # Make sure wandb config doesn't contain any CUDA tensors
            wandb_config = {k: v.cpu().item() if isinstance(v, t.Tensor) else v 
                          for k, v in wandb_config.items()}
            wandb_process = mp.Process(
                target=new_wandb_process,
                args=(wandb_config, log_queue, wandb_entity, wandb_project),
            )
            wandb_process.start()
            wandb_processes.append(wandb_process)

    # make save dirs, export config
    if save_dir is not None:
        save_dirs = [
            os.path.join(save_dir, f"trainer_{i}") for i in range(len(trainer_configs))
        ]
        for trainer, dir in zip(trainers, save_dirs):
            os.makedirs(dir, exist_ok=True)
            # save config
            config = {"trainer": trainer.config}
            try:
                config["buffer"] = data.config
            except:
                pass
            with open(os.path.join(dir, "config.json"), "w") as f:
                json.dump(config, f, indent=4)
    else:
        save_dirs = [None for _ in trainer_configs]

    if normalize_activations:
        norm_factor = get_norm_factor(data, steps=100)

        for trainer in trainers:
            trainer.config["norm_factor"] = norm_factor
            # Verify that all autoencoders have a scale_biases method
            trainer.ae.scale_biases(1.0)

    for step, act in enumerate(tqdm(data, total=steps)):

        act = act.to(dtype=autocast_dtype)

        if normalize_activations:
            act /= norm_factor

        if step >= steps:
            break

        # logging
        if (use_wandb or verbose) and step % log_steps == 0:
            log_stats(
                trainers, step, act, activations_split_by_head, transcoder, log_queues=log_queues, verbose=verbose
            )

        # saving
        if save_steps is not None and step in save_steps:
            for dir, trainer in zip(save_dirs, trainers):
                if dir is not None:

                    if normalize_activations:
                        # Temporarily scale up biases for checkpoint saving
                        trainer.ae.scale_biases(norm_factor)

                    if not os.path.exists(os.path.join(dir, "checkpoints")):
                        os.mkdir(os.path.join(dir, "checkpoints"))

                    checkpoint = {k: v.cpu() for k, v in trainer.ae.state_dict().items()}
                    t.save(
                        checkpoint,
                        os.path.join(dir, "checkpoints", f"ae_{step}.pt"),
                    )

                    if normalize_activations:
                        trainer.ae.scale_biases(1 / norm_factor)

        # training
        for i, trainer in enumerate(trainers):
            with autocast_context:
                # Pass the log queues to the update method
                current_queue = log_queues[i] if use_wandb and i < len(log_queues) else None
                trainer.update(step, act, log_queues=[current_queue] if current_queue else None)

    # save final SAEs
    for save_dir, trainer in zip(save_dirs, trainers):
        if normalize_activations:
            trainer.ae.scale_biases(norm_factor)
        if save_dir is not None:
            final = {k: v.cpu() for k, v in trainer.ae.state_dict().items()}
            t.save(final, os.path.join(save_dir, "ae.pt"))

    # Upload to HuggingFace if specified
    if hf_repo_out is not None:
        print(f"Uploading to HuggingFace repo: {hf_repo_out}")
        api = HfApi()
        
        # Try to create the repo if it doesn't exist
        try:
            api.create_repo(
                repo_id=hf_repo_out,
                repo_type="model",
                private=False,  # You might want to make this configurable
                exist_ok=True  # This ensures we don't get an error if the repo already exists
            )
            if verbose:
                print(f"Repository {hf_repo_out} is ready")
        except Exception as e:
            print(f"Error creating/accessing repository {hf_repo_out}: {str(e)}")
            return  # Exit if we can't create/access the repo
        
        for i, (save_dir, trainer) in enumerate(zip(save_dirs, trainers)):
            if save_dir is not None:
                # Create a folder for this specific trainer in the repo
                repo_path = f"{hf_repo_out}/trainer_{i}"
                
                # Save config and model files
                config_path = os.path.join(save_dir, "config.json")
                model_path = os.path.join(save_dir, "ae.pt")
                
                try:
                    # Upload config
                    api.upload_file(
                        path_or_fileobj=config_path,
                        path_in_repo=f"trainer_{i}/config.json",
                        repo_id=hf_repo_out,
                        repo_type="model",
                    )
                    
                    # Upload model
                    api.upload_file(
                        path_or_fileobj=model_path,
                        path_in_repo=f"trainer_{i}/ae.pt",
                        repo_id=hf_repo_out,
                        repo_type="model",
                    )
                    
                    if verbose:
                        print(f"Successfully uploaded trainer_{i} to {hf_repo_out}")
                        
                except Exception as e:
                    print(f"Error uploading trainer_{i} to HuggingFace: {str(e)}")
                    continue

    # Signal wandb processes to finish
    if use_wandb:
        for queue in log_queues:
            queue.put("DONE")
        for process in wandb_processes:
            process.join()