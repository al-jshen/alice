import math

import torch
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import LambdaLR

import wandb


def get_wandb_run(run_path: str):
    """Gets a W&B run config from the run path (entity/project/run_id)"""
    wandb.login()
    wandb_api = wandb.Api()
    run = wandb_api.run(run_path)
    return OmegaConf.create(run.config)


def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, min_learning_rate=0.0
):
    """
    Create a schedule with a linear warmup, cosine decay to `min_learning_rate`, then constant.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        num_warmup_steps (int): Number of warmup steps.
        num_training_steps (int): Total number of training steps (after which LR stays constant).
        min_learning_rate (float): Minimum LR value as an absolute LR.

    Returns:
        LambdaLR scheduler.
    """
    base_lr = optimizer.defaults["lr"]
    min_lr_multiplier = min_learning_rate / base_lr

    if min_lr_multiplier < 0.0 or min_lr_multiplier > 1.0:
        raise ValueError(
            f"min_learning_rate ({min_learning_rate}) must be between 0 and {base_lr}."
        )

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step < num_training_steps:
            # Cosine decay from 1.0 to min_lr_multiplier
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return cosine_decay * (1.0 - min_lr_multiplier) + min_lr_multiplier
        else:
            # Stay constant at min_lr_multiplier
            return min_lr_multiplier

    return LambdaLR(optimizer, lr_lambda)


def collate_dicts(dict_list):
    if not dict_list:
        return {}

    collated = {}
    keys = dict_list[0].keys()

    for key in keys:
        values = [d[key] for d in dict_list]

        if isinstance(values[0], list):
            collated[key] = sum(values, [])  # Flatten list of lists
        elif torch.is_tensor(values[0]):
            collated[key] = torch.cat(values, dim=0)
        else:
            collated[key] = values

    return collated
