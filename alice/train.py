import argparse
import os
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from alice.data.load_data import get_dataloader
from alice.models.transformer import Transformer
from alice.utils import get_cosine_schedule_with_warmup


def save_checkpoint(
    model, ema_model, optimizer, scheduler, global_step, config, save_dir="checkpoints"
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "global_step": global_step,
        "config": OmegaConf.to_container(config),
    }
    if ema_model is not None:
        checkpoint["ema_model_state_dict"] = ema_model.state_dict()
    torch.save(checkpoint, f"{save_dir}/checkpoint_{global_step}.pt")
    print(f"Checkpoint saved at global step {global_step}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a model with OmegaConf configuration."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration file.",
    )
    return parser.parse_args()


def setup(args):
    config = OmegaConf.load(args.config)

    defaults = OmegaConf.create(
        dict(
            # dataset
            dataset="jstet/quotes-500k",
            extra_chars='.,!?-;:"',
            min_length=15,
            max_length=300,
            cipher_type="cryptogram",
            cipher_kwargs={},
            test_cipher_type="cryptogram",
            test_cipher_kwargs={},
            num_workers=8,
            batch_size=64,
            test_size=0.025,
            loss_mask_punctuation=False,
            loss_mask_spaces=False,
            seed=0,
            # model
            dim=768,
            n_head=12,
            n_layer=12,
            attn_dropout_p=0.1,
            resid_dropout_p=0.1,
            ffn_dim_multiplier=None,
            ffn_dropout_p=0.1,
            drop_path=0.1,
            dynamic_embeddings=False,
            unique_decoding=True,
            embedding_n_layer=0,
            embedding_n_head=0,
            compile_model=False,
            use_ema=False,
            ema_decay=0.999,
            sinkhorn_decoding=False,
            sinkhorn_iters=10,
            sinkhorn_tau=1.0,
            sinkhorn_schedule="constant",
            sinkhorn_decay_steps=100_000,
            sinkhorn_min_tau=0.1,
            # training
            learning_rate=1e-4,
            warmup_steps=0,
            max_iterations=100000,
            min_learning_rate=1e-4,
            lr_scheduler="constant",
            weight_decay=0.1,
            beta1=0.9,
            beta2=0.95,
            epsilon=1e-5,
            epochs=100,
            validate=True,
            val_freq=1000,
            drop_last=False,
            bf16=True,
        )
    )
    config = OmegaConf.merge(defaults, config)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dl_train, dl_test, tok_mapping = get_dataloader(
        dataset=config.dataset,
        extra_chars=config.extra_chars,
        min_length=config.min_length,
        max_length=config.max_length,
        cipher_type=config.cipher_type,
        cipher_kwargs=OmegaConf.to_object(config.cipher_kwargs),
        test_cipher_type=config.test_cipher_type,
        test_cipher_kwargs=OmegaConf.to_object(config.test_cipher_kwargs),
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        test_size=config.test_size,
        pin_memory=device == "cuda",
        drop_last=config.drop_last,
        loss_mask_punctuation=config.loss_mask_punctuation,
        loss_mask_spaces=config.loss_mask_spaces,
        seed=config.seed,
    )

    model = Transformer(
        vocab_size=len(tok_mapping) + 1,  # extra one for padding
        dim=config.dim,
        n_head=config.n_head,
        n_layer=config.n_layer,
        attn_dropout_p=config.attn_dropout_p,
        resid_dropout_p=config.resid_dropout_p,
        ffn_dim_multiplier=config.ffn_dim_multiplier,
        ffn_dropout_p=config.ffn_dropout_p,
        drop_path=config.drop_path,
        dynamic_embeddings=config.dynamic_embeddings,
        unique_decoding=config.unique_decoding,
        embedding_n_layer=config.embedding_n_layer,
        embedding_n_head=config.embedding_n_head,
        sinkhorn_decoding=config.sinkhorn_decoding,
        sinkhorn_tau=config.sinkhorn_tau,
        sinkhorn_iters=config.sinkhorn_iters,
        sinkhorn_schedule=config.sinkhorn_schedule,
        sinkhorn_decay_steps=config.sinkhorn_decay_steps,
        sinkhorn_min_tau=config.sinkhorn_min_tau,
    ).to(device)

    if config.compile_model:
        model = torch.compile(model)

    if config.use_ema:
        ema_model = optim.swa_utils.AveragedModel(
            model,
            multi_avg_fn=optim.swa_utils.get_ema_multi_avg_fn(config.ema_decay),
            use_buffers=True,
        )
    else:
        ema_model = None

    optimizer_kwargs = dict(
        betas=(getattr(config, "beta1", 0.9), getattr(config, "beta2", 0.999)),
        eps=getattr(config, "epsilon", 1e-8),
        lr=config.learning_rate,
        weight_decay=getattr(config, "weight_decay", 0.01),
    )
    optimizer = optim.AdamW(
        model.parameters(),
        **optimizer_kwargs,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.max_iterations,
        min_learning_rate=config.min_learning_rate
        if config.lr_scheduler == "cosine"
        else config.learning_rate,
    )

    if getattr(config, "load_ckpt", None):
        checkpoint = torch.load(config.load_ckpt, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if ema_model is not None:
            ema_model.load_state_dict(checkpoint.get("ema_model_state_dict", {}))
        global_step = checkpoint["global_step"]
        assert hasattr(config, "wandb_id"), (
            "wandb_id must be set in the config when loading a checkpoint."
        )
    else:
        global_step = 0

    if config.wandb_enabled:
        run = wandb.init(
            entity=getattr(config, "wandb_entity", None),
            project=getattr(config, "wandb_project", None),
            group=getattr(config, "wandb_group", None),
            notes=getattr(config, "wandb_notes", None),
            id=getattr(config, "wandb_id", None),
            resume="allow",
            config=OmegaConf.to_container(config),
            mode=getattr(config, "wandb_mode", "online"),
        )
        run.watch(model, log="all", log_freq=config.wandb_log_freq)
    else:
        run = None

    checkpoint_dir = config.checkpoint_dir + run.id if run else config.checkpoint_dir

    os.makedirs(checkpoint_dir, exist_ok=True)

    return dict(
        model=model,
        ema_model=ema_model,
        dl_train=dl_train,
        dl_test=dl_test,
        tok_mapping=tok_mapping,
        device=device,
        run=run,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        checkpoint_dir=checkpoint_dir,
        global_step=global_step,
    )


def train(setup_dict):
    torch.set_float32_matmul_precision("high")

    model = setup_dict["model"]
    ema_model = setup_dict["ema_model"]
    dl_train = setup_dict["dl_train"]
    dl_test = setup_dict["dl_test"]
    tok_mapping = setup_dict["tok_mapping"]
    device = setup_dict["device"]
    run = setup_dict["run"]
    optimizer = setup_dict["optimizer"]
    scheduler = setup_dict["scheduler"]
    config = setup_dict["config"]
    checkpoint_dir = setup_dict["checkpoint_dir"]
    global_step = setup_dict["global_step"]

    mean_times = {
        "data_time": 0.0,
        "transfer_time": 0.0,
        "forward_time": 0.0,
        "backward_time": 0.0,
    }
    if ema_model is not None:
        mean_times["ema_time"] = 0.0

    grad_scaler = torch.amp.GradScaler(device=device, enabled=config.bf16)

    for _ in range(config.epochs):
        model.train()

        end_time = time.time()
        for batch in (pbar := tqdm(dl_train)):
            start_time = time.time()

            encrypted_tok = batch["encrypted_tok"].to(device)
            text_tok = batch["text_tok"].to(device)
            mask = batch["mask"].to(device)
            loss_mask = batch["loss_mask"].to(device)

            post_transfer_time = time.time()

            optimizer.zero_grad()

            with torch.amp.autocast(
                device_type=device, enabled=config.bf16, dtype=torch.bfloat16
            ):
                pred = model(
                    encrypted_tok, mask=mask[:, None, None, :], step=global_step
                )
                loss = F.cross_entropy(pred[loss_mask], text_tok[loss_mask])

            post_forward_time = time.time()

            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            scheduler.step()

            post_backward_time = time.time()

            if ema_model is not None:
                ema_model.update_parameters(model)

                post_ema_time = time.time()

            mean_times["data_time"] += start_time - end_time
            mean_times["transfer_time"] += post_transfer_time - start_time
            mean_times["forward_time"] += post_forward_time - post_transfer_time
            mean_times["backward_time"] += post_backward_time - post_forward_time
            if ema_model is not None:
                mean_times["ema_time"] += post_ema_time - post_backward_time

            pbar.set_description(f"train loss: {loss.item():.4f}")

            if run and global_step % config.wandb_log_freq == 0:
                mean_times = {
                    k: v / config.wandb_log_freq for k, v in mean_times.items()
                }
                log_dict = {
                    "global_step": global_step,
                    "train_loss": loss.item(),
                } | mean_times
                if config.log_lr:
                    log_dict["learning_rate"] = optimizer.param_groups[0]["lr"]
                run.log(log_dict)
                mean_times = {k: 0.0 for k in mean_times}

            if global_step % config.save_freq == 0:
                save_checkpoint(
                    model,
                    ema_model,
                    optimizer,
                    scheduler,
                    global_step,
                    config,
                    save_dir=checkpoint_dir,
                )

            global_step += 1

            if config.validate and global_step % config.val_freq == 0:
                model.eval()

                for m, mn in zip([model, ema_model], ["model", "ema_model"]):
                    if m is None:
                        continue

                    val_loss = 0.0
                    val_batches = 0

                    val_total_tok = 0
                    val_wrong_toks = 0

                    for batch in (pbar := tqdm(dl_test)):
                        encrypted_tok = batch["encrypted_tok"].to(device)
                        text_tok = batch["text_tok"].to(device)
                        mask = batch["mask"].to(device)
                        loss_mask = batch["loss_mask"].to(device)

                        with torch.no_grad():
                            pred = m(encrypted_tok, mask=mask[:, None, None, :])
                            loss = F.cross_entropy(pred[loss_mask], text_tok[loss_mask])

                            out = torch.argmax(pred, dim=-1)
                            wrong_toks = ((out != text_tok) & loss_mask).sum().item()
                            total_toks = loss_mask.sum().item()

                        val_loss += loss.item()
                        val_batches += 1
                        val_total_tok += total_toks
                        val_wrong_toks += wrong_toks

                        pbar.set_description(f"{mn} | val loss: {loss.item():.4f}")

                    if run:
                        run.log(
                            {
                                "global_step": global_step,
                                f"{mn}_val_loss": val_loss / val_batches,
                                f"{mn}_val_accuracy": 1
                                - (val_wrong_toks / val_total_tok),
                            }
                        )

                model.train()

            end_time = time.time()

    if run:
        run.finish()


if __name__ == "__main__":
    args = parse_args()
    setup_dict = setup(args)
    train(setup_dict)
