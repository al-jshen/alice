import os

import hydra
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from omegaconf import DictConfig

from alice.data.load_data import get_dataloader
from alice.models.probes import MultiLayerProbe
from alice.models.transformer import Transformer
from alice.utils import get_wandb_run


def save_checkpoint(probes, global_step, config, save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = {
        "probes_state_dict": probes.state_dict(),
        "global_step": global_step,
        "config": config,
    }
    torch.save(checkpoint, f"{save_dir}/checkpoint_step_{global_step}.pt")
    print(f"Checkpoint saved at step {global_step}")


@torch.no_grad()
def get_activations(model, tok, mask=None):
    x = model.embeddings(tok)  # BxNxD floats

    model.freqs_cis = model.freqs_cis.to(
        tok.device
    )  # ensure freqs_cis is on the same device
    freqs_cis = model.freqs_cis[: x.shape[1]]

    activations = []
    activations.append(x)
    for block in model.blocks:
        x = block(x, freqs_cis=freqs_cis, mask=mask)  # BxNxD floats
        activations.append(x)

    return activations  # (num_layers+1)xBxNxD floats


def setup(
    cfg: DictConfig,
    # full_wandb_id,
    # checkpoint_path,
    # probe_type,
    # probe_hidden_dim=None,
    # lr=1e-3,
    # wandb_entity=None,
    # wandb_project=None,
    # wandb_group=None,
    # wandb_notes=None,
    # wandb_log_freq=100,
):
    # print all config values and dtypes
    print("[setup] Configuration:")
    for k, v in cfg.items():
        print(f"  {k} ({type(v)}): {v}")
    print("")

    config = get_wandb_run(cfg.full_wandb_id)
    print(f"[setup] Loaded run config for {cfg.full_wandb_id}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] Device: {device}")

    dl_train, dl_test, tok_mapping = get_dataloader(
        dataset=config.data.dataset,
        extra_chars=config.data.extra_chars,
        min_length=config.data.min_length,
        max_length=config.data.max_length,
        cipher_type=config.data.cipher_type,
        cipher_kwargs={},  # config.data.get("cipher_kwargs", {}),
        test_cipher_type=config.data.test_cipher_type,
        test_cipher_kwargs={},  # config.data.get("test_cipher_kwargs", {}),
        num_workers=config.data.num_workers,
        batch_size=config.data.batch_size,
        test_size=config.data.test_size,
        pin_memory=device == "cuda",
        drop_last=config.data.drop_last,
        loss_mask_punctuation=config.data.loss_mask_punctuation,
        loss_mask_spaces=config.data.loss_mask_spaces,
        seed=config.data.seed,
    )

    # Basic data loader logging
    try:
        train_len = len(dl_train)
        test_len = len(dl_test)
    except TypeError:
        train_len = "unknown"
        test_len = "unknown"
    print(
        f"[setup] Data: dataset={config.data.dataset}, cipher={config.data.cipher_type} -> test_cipher={config.data.test_cipher_type}"
    )
    print(
        f"[setup] Dataloaders: train_batches={train_len}, test_batches={test_len}, batch_size={config.data.batch_size}, num_workers={config.data.num_workers}"
    )

    model = Transformer(
        vocab_size=len(tok_mapping) + 1,  # extra one for padding
        dim=config.model.dim,
        n_head=config.model.n_head,
        n_layer=config.model.n_layer,
        attn_dropout_p=config.model.attn_dropout_p,
        resid_dropout_p=config.model.resid_dropout_p,
        ffn_dim_multiplier=None,
        ffn_dropout_p=config.model.ffn_dropout_p,
        drop_path=config.model.drop_path,
        dynamic_embeddings=config.model.dynamic_embeddings,
        unique_decoding=config.model.unique_decoding,
        embedding_n_layer=config.model.embedding_n_layer,
        embedding_n_head=config.model.embedding_n_head,
    ).to(device)

    # Basic model logging
    print(
        f"[setup] Model: Transformer layers={config.model.n_layer}, dim={config.model.dim}, heads={config.model.n_head}, vocab_size={len(tok_mapping) + 1}"
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[setup] Model parameters: {total_params:,}")

    print(f"[setup] Loading checkpoint from {cfg.checkpoint_path}")
    ckpt = torch.load(cfg.checkpoint_path, weights_only=False, mmap=True)
    print(model.load_state_dict(ckpt["model_state_dict"], strict=False))
    model.eval()

    probes = MultiLayerProbe(
        probe=cfg.probe_type,
        num_probes=len(model.blocks) + 1,
        input_dim=config.model.dim,
        hidden_dim=cfg.probe_hidden_dim if cfg.probe_type != "linear" else None,
        output_dim=len(tok_mapping) + 1,
    ).to(device)

    print(
        f"[setup] Probes: type={cfg.probe_type}, num_probes={len(model.blocks) + 1}, hidden_dim={cfg.probe_hidden_dim}, output_dim={len(tok_mapping) + 1}"
    )

    print(
        f"[setup] Probes parameters (per layer): {sum(p.numel() for p in probes.probes[0].parameters()):,}"
    )

    optimizer = optim.AdamW(probes.parameters(), lr=cfg.lr)
    print(f"[setup] Optimizer: AdamW lr={cfg.lr}")

    log_config = dict(
        full_wandb_id=cfg.full_wandb_id,
        checkpoint_path=cfg.checkpoint_path,
        probe_type=cfg.probe_type,
        probe_hidden_dim=cfg.probe_hidden_dim,
        lr=cfg.lr,
    )

    print(f"[setup] Weights & Biases enabled: {config.wandb.wandb_enabled}")
    if config.wandb.wandb_enabled:
        run = wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            group=cfg.wandb_group,
            notes=cfg.wandb_notes,
            resume="allow",
            config=log_config,
        )
        print(
            f"[setup] WandB initialized: project={cfg.wandb_project}, group={cfg.wandb_group}, entity={cfg.wandb_entity}"
        )
        run.watch(model, log="all", log_freq=cfg.wandb_log_freq)
    else:
        run = None

    return dict(
        model=model,
        probes=probes,
        optimizer=optimizer,
        dl_train=dl_train,
        dl_test=dl_test,
        tok_mapping=tok_mapping,
        device=device,
        cfg=cfg,
        run=run,
    )


def train(setup_dict):
    model = setup_dict["model"]
    probes = setup_dict["probes"]
    optimizer = setup_dict["optimizer"]
    dl_train = setup_dict["dl_train"]
    dl_test = setup_dict["dl_test"]
    device = setup_dict["device"]
    cfg = setup_dict["cfg"]
    run = setup_dict["run"]

    print("[train] Starting training...")
    # debug by loading a single item from the dataloader
    # batch = next(iter(dl_train))
    # print("[train] Loaded test batch, sample batch keys:", batch.keys())

    global_step = 0

    for epoch in range(cfg.epochs):  # number of epochs
        model.eval()
        probes.train()

        total_loss = [0.0] * len(probes.probes)
        track_batches = 0
        for batch_idx, batch in enumerate(dl_train):
            encrypted_tok = batch["encrypted_tok"].to(device)
            text_tok = batch["text_tok"].to(device)
            mask = batch["mask"].to(device)
            loss_mask = batch["loss_mask"].to(device)

            optimizer.zero_grad()

            # print("getting activations")
            activations = get_activations(
                model, encrypted_tok, mask=mask[:, None, None, :]
            )

            # print("getting logits from probes")
            logits_per_layer = probes(activations)  # list of (B, N, vocab_size) logits

            # print("computing loss")
            losses = [
                F.cross_entropy(logits[loss_mask], text_tok[loss_mask])
                for logits in logits_per_layer
            ]
            loss = sum(losses) / len(logits_per_layer)

            # print("backpropagating")
            loss.backward()
            optimizer.step()

            for layer_idx in range(len(losses)):
                total_loss[layer_idx] += losses[layer_idx].item()
            track_batches += 1

            if batch_idx % int(cfg.print_freq) == 0:
                avg_loss_per_layer = [loss / track_batches for loss in total_loss]
                for layer_idx, avg_loss in enumerate(avg_loss_per_layer):
                    print(
                        f"Epoch [{epoch + 1}], Step [{batch_idx + 1}], Layer [{layer_idx}], Loss: {avg_loss:.4f}"
                    )
                print("\n")

            if batch_idx % int(cfg.wandb_log_freq) == 0:
                avg_loss_per_layer = [loss / track_batches for loss in total_loss]
                for layer_idx, avg_loss in enumerate(avg_loss_per_layer):
                    wandb.log(
                        {f"train/layer_{layer_idx}_loss": avg_loss},
                        step=epoch * len(dl_train) + batch_idx,
                    )
                total_loss = [0.0] * len(probes.probes)
                track_batches = 0

            global_step += 1

            # Save checkpoint at the end of each epoch or based on the frequency
            if global_step % int(cfg.checkpoint_freq) == 0:
                checkpoint_dir = os.path.join(cfg.save_dir, run.id)
                save_checkpoint(
                    probes, global_step, config=cfg, save_dir=checkpoint_dir
                )

        # check point at end of epoch
        checkpoint_dir = os.path.join(cfg.save_dir, run.id)
        save_checkpoint(probes, global_step, config=cfg, save_dir=checkpoint_dir)

        # Evaluation
        model.eval()
        probes.eval()

        total_wrong = [0] * len(probes.probes)
        total_count = [0] * len(probes.probes)

        with torch.no_grad():
            for batch in dl_test:
                encrypted_tok = batch["encrypted_tok"].to(device)
                text_tok = batch["text_tok"].to(device)
                mask = batch["mask"].to(device)
                loss_mask = batch["loss_mask"].to(device)

                activations = get_activations(
                    model, encrypted_tok, mask=mask[:, None, None, :]
                )
                logits_per_layer = probes(activations)

                for layer_idx, logits in enumerate(logits_per_layer):
                    out = torch.argmax(logits, dim=-1)
                    # correct = ((out == text_tok) & (text_tok != 0)).sum().item()
                    # total = (text_tok != 0).sum().item()  # ignore padding

                    wrong_toks = ((out != text_tok) & loss_mask).sum().item()
                    total_toks = loss_mask.sum().item()
                    total_wrong[layer_idx] += wrong_toks
                    total_count[layer_idx] += total_toks

        for layer_idx in range(len(probes.probes)):
            accuracy = 1 - (total_wrong[layer_idx] / total_count[layer_idx])
            print(
                f"Epoch [{epoch + 1}], Layer [{layer_idx}], Test Accuracy: {accuracy:.4f}"
            )
            wandb.log({f"test/layer_{layer_idx}_accuracy": accuracy}, step=global_step)


@hydra.main(version_base=None, config_path="../configs", config_name="train_probes")
def main(cfg: DictConfig):
    setup_dict = setup(cfg)
    train(setup_dict)


if __name__ == "__main__":
    main()
