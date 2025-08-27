import argparse
import string

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from alice.data.load_data import get_dataloader
from alice.models.modules import harden_permutation
from alice.models.transformer import Transformer, average_embeddings_by_id
from alice.utils import get_wandb_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_type",
        type=str,
        default="cryptogram",
        choices=["cryptogram", "multilingual", "bijective", "dynamic"],
        help="Type of run to interpret",
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    run_type = args.run_type

    run_settings = {
        "cryptogram": {
            "run_name": "<wandb_run_id>",
            "file_name": "<checkpoint_path.pt>",
        },
        "multilingual": {
            "run_name": "<wandb_run_id>",
            "file_name": "<checkpoint_path.pt>",
        },
        "bijective": {
            "run_name": "<wandb_run_id>",
            "file_name": "<checkpoint_path.pt>",
        },
        "dynamic": {"run_name": "<wandb_run_id>", "file_name": "<checkpoint_path.pt>"},
    }[run_type]

    run_name = run_settings["run_name"]
    file_name = run_settings["file_name"]

    # load model from checkpoint using state_dict
    checkpoint = torch.load(file_name, map_location=torch.device(device))
    config = get_wandb_run(f"<wandb_entity>/<wandb_run_path>/{run_name}")

    _, dl_test, tok_mapping = get_dataloader(
        dataset=config.dataset,
        extra_chars=config.extra_chars,
        min_length=config.min_length,
        max_length=config.max_length,
        cipher_type=config.cipher_type,
        cipher_kwargs=config.get("cipher_kwargs", {}),
        test_cipher_type=config.test_cipher_type,
        test_cipher_kwargs=config.get("test_cipher_kwargs", {}),
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        test_size=config.test_size,
        pin_memory=device == "cuda",
        drop_last=config.drop_last,
        loss_mask_punctuation=config.loss_mask_punctuation,
        loss_mask_spaces=config.loss_mask_spaces,
        seed=config.seed,
    )
    if run_type == "bijective":
        model = Transformer(
            vocab_size=len(tok_mapping) + 1,  # extra one for padding
            dim=config.dim,
            n_head=config.n_head,
            n_layer=config.n_layer,
            attn_dropout_p=config.attn_dropout_p,
            resid_dropout_p=config.resid_dropout_p,
            ffn_dim_multiplier=None,
            ffn_dropout_p=config.ffn_dropout_p,
            drop_path=config.drop_path,
            dynamic_embeddings=config.dynamic_embeddings,
            unique_decoding=config.unique_decoding,
            embedding_n_layer=config.embedding_n_layer,
            embedding_n_head=config.embedding_n_head,
            sinkhorn_decoding=config.sinkhorn_decoding,
            sinkhorn_tau=config.sinkhorn_tau,
            sinkhorn_iters=config.sinkhorn_iters,
            sinkhorn_schedule="constant",
        ).to(device)
    else:
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
        ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    print("n_layer:", config.n_layer)

    @torch.no_grad()
    def interp_forward(model, tok, mask=None):
        x = model.embeddings(tok)  # BxNxD floats

        model.freqs_cis = model.freqs_cis.to(
            tok.device
        )  # ensure freqs_cis is on the same device
        freqs_cis = model.freqs_cis[: x.shape[1]]

        temp = []
        temp.append(x)
        for block in model.blocks:
            x = block(x, freqs_cis=freqs_cis, mask=mask)  # BxNxD floats
            temp.append(x)

        intermediate_outputs = []
        for x in temp:
            x = model.norm(x)  # BxNxD floats

            # unique_decoding
            x = average_embeddings_by_id(
                x, tok, max_vocab_size=model.vocab_size
            )  # pool embeddings by token ID, still BxNxD floats

            x = model.debed(x)  # BxNxV floats
            intermediate_outputs.append(x)

        return intermediate_outputs  # n_layersxBxNxV floats, distribution over vocab per token

    @torch.no_grad()
    def predict(model, batch, tok_mapping, return_P=False):
        tok_mapping_inverse = {v: k for k, v in tok_mapping.items()}

        def decode_back(txt):
            return "".join([tok_mapping_inverse.get(i.item(), "") for i in txt])

        device = next(model.parameters()).device
        encrypted_tok = batch["encrypted_tok"].to(device)
        text_tok = batch["text_tok"].to(device) if "text_tok" in batch.keys() else None
        mask = batch["mask"].to(device)
        loss_mask = batch["loss_mask"].to(device)

        pred = model(
            encrypted_tok, mask=mask[:, None, None, :], return_permutation=return_P
        )
        if return_P:
            pred, P = pred
        out = torch.argmax(pred, dim=-1)

        wrong = ((out != text_tok) & loss_mask).cpu()
        where_wrong = torch.where(wrong)
        num_wrong = wrong.sum(axis=1)
        num_total = loss_mask.sum(axis=1).cpu()

        bs = len(encrypted_tok)

        decoded_text = (
            [decode_back(text_tok[idx]) for idx in range(bs)]
            if text_tok is not None
            else None
        )
        decoded_encrypted = [decode_back(encrypted_tok[idx]) for idx in range(bs)]
        decoded_out = [decode_back(out[idx]) for idx in range(bs)]

        decoded_out_clean = [
            "".join(
                [
                    decoded_encrypted[idx][i]
                    if (~loss_mask[idx])[i].item()
                    else decoded_out[idx][i]
                    for i in range(len(decoded_encrypted[idx]))
                ]
            )
            for idx in range(bs)
        ]

        return dict(
            orig_text=decoded_text,
            encrypted_text=decoded_encrypted,
            predicted_text=decoded_out_clean,
            # where_wrong=where_wrong,
            wrong=wrong,
            num_wrong=num_wrong,
            num_total=num_total,
            seq_lens=torch.tensor([len(i) for i in decoded_text]),
            P=P.cpu() if return_P else None,
            cipher=batch["cipher"],
        )

    def mapping_to_matrix(mapping, ciphertext):
        mat = np.zeros((26, 26))
        mask = np.ones((26, 26))
        unobserved = set(string.ascii_uppercase).difference(
            set(ciphertext).intersection(set(string.ascii_uppercase))
        )
        for i in range(26):
            map_letter = mapping[string.ascii_uppercase[i]]

            if map_letter in unobserved:
                # pass
                mask[:, string.ascii_uppercase.index(map_letter)] = 0.0
                mask[i, :] = 0.0

            mat[i, string.ascii_uppercase.index(map_letter)] = 1

        return mat, mask

    @torch.no_grad()
    def interp_forward_bij(model, tok, mask=None, return_permutation=False):
        x = model.embeddings(tok)  # BxNxD floats

        model.freqs_cis = model.freqs_cis.to(
            tok.device
        )  # ensure freqs_cis is on the same device
        freqs_cis = model.freqs_cis[: x.shape[1]]

        temp = []
        temp.append(x)
        for block in model.blocks:
            x = block(x, freqs_cis=freqs_cis, mask=mask)  # BxNxD floats
            temp.append(x)

        intermediate_outputs = []
        Ps = []

        for x in temp:
            x = model.norm(x)  # BxNxD floats

            # unique_decoding
            x = average_embeddings_by_id(
                x, tok, max_vocab_size=model.vocab_size
            )  # pool embeddings by token ID, still BxNxD floats

            # bij model output head
            if model.sinkhorn_decoding:
                query = model.output_query.unsqueeze(0).expand(x.shape[0], -1, -1)
                x = F.scaled_dot_product_attention(
                    query, x, x, attn_mask=mask.squeeze(1) if mask is not None else None
                )  # squeeze to get rid of head dim
                x = model.debed(x)  # BxVxV floats, log permutation matrix over vocab

                P = (
                    torch.stack(
                        [harden_permutation(i) for i in x.cpu().detach().numpy()]
                    )
                    .float()
                    .to(x.device)
                )

                one_hot = F.one_hot(tok, num_classes=model.vocab_size).float()

                x = torch.einsum(
                    "bvw,bnv->bnw", P, one_hot
                )  # BxNxV floats, logits over vocab per token
                # x = torch.einsum("bnv,bvw->bnw", one_hot, P.transpose(1, 2))
            else:
                x = model.debed(x)  # BxNxV floats, logits over vocab per token

            outputs = [x]

            if return_permutation:
                Ps.append(P)

            intermediate_outputs.append(x)

            # x = model.debed(x)  # BxNxV floats
            # intermediate_outputs.append(x)

        return (
            intermediate_outputs,
            Ps,
        )  # n_layersxBxNxV floats, distribution over vocab per token

    # calculate the error rate for each layer
    n_layer = config.n_layer + 1
    val_total_tok = 0
    val_wrong_toks = np.zeros(n_layer)
    tok_mapping_inverse = {v: k for k, v in tok_mapping.items()}
    return_P = False
    Ps = None
    print(model.sinkhorn_decoding, run_type)
    if run_type == "bijective":
        return_P = True
        Ps = []

    def decode_back(txt):
        return "".join([tok_mapping_inverse.get(i.item(), "") for i in txt])

    for batch in tqdm(dl_test):
        encrypted_tok = batch["encrypted_tok"].to(device)
        text_tok = batch["text_tok"].to(device)
        mask = batch["mask"].to(device)
        loss_mask = batch["loss_mask"].to(device)
        bs = len(encrypted_tok)

        if run_type == "bijective":
            intermediate_outputs = interp_forward_bij(
                model,
                encrypted_tok,
                mask=mask[:, None, None, :],
                return_permutation=return_P,
            )
            if return_P:
                intermediate_outputs, P = (
                    intermediate_outputs[0],
                    intermediate_outputs[1],
                )  # LxBxNxV, LxBxVxV
                Ps.append(P)
        else:
            intermediate_outputs = interp_forward(
                model, encrypted_tok, mask=mask[:, None, None, :]
            )
        outs = [
            torch.argmax(pred, dim=-1) for pred in intermediate_outputs
        ]  # n_layersxBxN
        # sum across sequence
        wrong_toks = [
            torch.sum(((out != text_tok) & loss_mask)) for out in outs
        ]  # n_layers
        total_toks = torch.sum(loss_mask)

        # print outputs to text file
        with open(f"interp_outputs_{run_type}.txt", "a") as f:
            decoded_text = decode_back(text_tok[0])
            decoded_encrypted = decode_back(encrypted_tok[0])
            f.write(f"true text: {decoded_text}\n")
            f.write(f"encrypted text: {decoded_encrypted}\n")

            for layer in range(len(outs)):
                decoded_out = decode_back(outs[layer][0])
                decoded_out_clean = "".join(
                    [
                        decoded_encrypted[i]
                        if (~loss_mask[0])[i].item()
                        else decoded_out[i]
                        for i in range(len(decoded_encrypted))
                    ]
                )
                f.write(f"outputs after layer {layer}: {decoded_out_clean}\n")

        # calculate the error rate for each layer
        val_wrong_toks += torch.stack(wrong_toks, dim=0).cpu().numpy()  # L
        val_total_tok += total_toks.cpu().numpy()

    # throw out last batch
    if Ps is not None:
        Ps = Ps[:-1]  # len(dl_test)xLxBxVxV
        Ps = torch.stack([torch.stack(p) for p in Ps])
        print(Ps.shape)

    # calculate the error rate for each layer
    error_rates = val_wrong_toks / val_total_tok
    # error_bars = np.std(val_wrong_toks / val_total_tok)
    print("error rates:", error_rates)

    layers = np.arange(len(outs))

    plt.figure()
    plt.plot(layers, error_rates, marker="o")
    # plt.errorbar(layers, error_rates, yerr=error_bars, fmt='o')

    plt.title(f"{run_type} error rate by layer")
    plt.xlabel("Layer")
    plt.ylabel("Error Rate")
    plt.xticks(layers)
    plt.grid()
    # plt.yscale('log')
    if len(layers) == 13 or len(layers) == 11 or run_type == "bijective_small":
        run_type = f"{run_type}_withemb"
    plt.savefig(f"layer_error_rate_{run_type}.pdf")

    # save the error rates and bars for later plotting
    np.save(f"layer_error_rates_{run_type}.npy", error_rates)
    # np.save(f'layer_error_bars_{run_type}.npy', error_bars)


if __name__ == "__main__":
    main()

