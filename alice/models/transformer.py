import math
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import (
    Attention,
    DropPath,
    FeedForward,
    RMSNorm,
    gumbel_sinkhorn,
    harden_permutation,
    linear_init,
    precompute_freqs_cis,
)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        n_head: int = 8,
        attn_dropout_p: float = 0.0,
        resid_dropout_p: float = 0.0,
        ffn_dim_multiplier: Optional[float] = None,
        ffn_dropout_p: float = 0.0,
        norm_eps: float = 1e-5,
        multiple_of: int = 256,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.attention = Attention(
            dim=dim,
            n_head=n_head,
            attn_dropout_p=attn_dropout_p,
            resid_dropout_p=resid_dropout_p,
        )
        self.feed_forward = FeedForward(
            dim=dim,
            ffn_dim_multiplier=ffn_dim_multiplier,
            ffn_dropout_p=ffn_dropout_p,
            multiple_of=multiple_of,
        )
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ):
        attn = self.attention(
            self.attention_norm(x), freqs_cis, mask, return_attn=return_attn
        )
        if return_attn:
            attn, attn_weights = attn
        h = x + self.drop_path(attn)
        out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
        if return_attn:
            return out, attn_weights
        else:
            return out


def average_embeddings_by_id(
    embeddings: torch.Tensor, ids: torch.Tensor, max_vocab_size: Optional[int] = None
) -> torch.Tensor:
    """
    Args:
        embeddings: (B, N, D) float tensor
        ids: (B, N) int tensor, with values from a small vocabulary (e.g., 0..V-1)
    Returns:
        Averaged embeddings per unique ID (within each batch), same shape as embeddings
    """
    B, N, D = embeddings.shape
    V = max_vocab_size or (ids.max().item() + 1)

    # Flatten batch and position into a single dimension
    flat_ids = ids + (torch.arange(B, device=ids.device).unsqueeze(1) * V)  # (B, N)
    flat_ids = flat_ids.view(-1)  # (B*N,)
    flat_emb = embeddings.view(B * N, D)  # (B*N, D)

    # Count how many times each ID appears
    counts = torch.zeros(B * V, device=embeddings.device).scatter_add_(
        0, flat_ids, torch.ones_like(flat_ids, dtype=embeddings.dtype)
    )  # (B*V,)

    # Sum embeddings for each unique ID
    summed = torch.zeros(B * V, D, device=embeddings.device).scatter_add_(
        0, flat_ids.unsqueeze(-1).expand(-1, D), flat_emb
    )  # (B*V, D)

    # Avoid division by zero
    counts = counts.clamp(min=1.0).unsqueeze(1)  # (B*V, 1)
    averaged = summed / counts  # (B*V, D)

    # Gather the averaged embedding for each (B, N)
    averaged_emb = averaged[flat_ids]  # (B*N, D)
    return averaged_emb.view(B, N, D)


class EmbeddingHypernet(nn.Module):
    """A hypernetwork that generates embeddings for a given vocabulary size.
    Given some input tokens BxN ints, a vocab size V, and an embedding dimension D,
    it generates a BxVxD tensor of embeddings which can be used to look up embeddings for tokens.
    """

    def __init__(
        self,
        vocab_size: int = 27,
        embedding_dim: int = 256,
        n_layer: int = 4,
        n_head: int = 8,
        attn_dropout_p: float = 0.0,
        resid_dropout_p: float = 0.0,
        ffn_dropout_p: float = 0.0,
        norm_eps: float = 1e-5,
        drop_path: float = 0.0,
        ffn_dim_multiplier: Optional[float] = None,
        max_seq_len: int = 1024,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = embedding_dim
        self.dummy_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.query = nn.Parameter(torch.randn(vocab_size, embedding_dim))  # BxVxD

        drop_path_rates = torch.linspace(0, drop_path, n_layer).tolist()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embedding_dim,
                    n_head=n_head,
                    ffn_dim_multiplier=ffn_dim_multiplier,
                    attn_dropout_p=attn_dropout_p,
                    resid_dropout_p=resid_dropout_p,
                    ffn_dropout_p=ffn_dropout_p,
                    norm_eps=norm_eps,
                    multiple_of=embedding_dim // n_head,
                    drop_path=drop_path_rates[i],
                )
                for i in range(n_layer)
            ]
        )

        self.freqs_cis = precompute_freqs_cis(
            seq_len=max_seq_len,
            n_elem=embedding_dim // n_head,
            base=rope_base,
            cls_token_num=0,
        )

    def generate_embeddings(self, tok: torch.Tensor, mask=None) -> torch.Tensor:
        # tok is BxN ints, we want to generate BxVxD embeddings
        B, N = tok.shape
        x = self.dummy_embedding(tok)  # BxNxD floats
        freqs_cis = self.freqs_cis[:N].to(tok)  # use only the first N frequencies
        for block in self.blocks:
            x = block(x, freqs_cis=freqs_cis, mask=mask)  # do some processing
        # cross attention with learnable query to get BxVxD embeddings
        query = self.query.unsqueeze(0).expand(B, -1, -1)  # BxVxD
        x = F.scaled_dot_product_attention(
            query, x, x, attn_mask=mask.squeeze(1) if mask is not None else None
        )  # BxVxD
        return x  # embedding matrix for each batch

    def forward(self, tok: torch.Tensor, mask=None) -> torch.Tensor:
        """Generate embeddings for the input tokens."""
        embeddings = self.generate_embeddings(tok, mask=mask)
        return torch.gather(embeddings, 1, tok.unsqueeze(-1).expand(-1, -1, self.dim))


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 27,
        dim: int = 256,
        n_head: int = 8,
        n_layer: int = 8,
        attn_dropout_p: float = 0.0,
        resid_dropout_p: float = 0.0,
        ffn_dim_multiplier: Optional[float] = None,
        ffn_dropout_p: float = 0.0,
        norm_eps: float = 1e-5,
        multiple_of: int = 256,
        drop_path: float = 0.0,
        dynamic_embeddings: bool = False,
        unique_decoding: bool = False,
        sinkhorn_decoding: bool = False,
        sinkhorn_tau: float = 1.0,
        sinkhorn_iters: int = 5,
        sinkhorn_schedule: Literal["linear", "cosine", "constant"] = "constant",
        sinkhorn_decay_steps: int = 100_000,
        sinkhorn_min_tau: float = 0.1,
        max_seq_len: int = 1024,
        rope_base: float = 10000.0,
        embedding_n_layer: int = 4,
        embedding_n_head: int = 8,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        if not dynamic_embeddings:
            self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=dim)
        else:
            self.embeddings = EmbeddingHypernet(
                vocab_size=vocab_size,
                embedding_dim=dim,
                n_layer=embedding_n_layer,
                n_head=embedding_n_head,
                max_seq_len=max_seq_len,
                rope_base=rope_base,
            )

        drop_path_rates = torch.linspace(0, drop_path, n_layer).tolist()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    n_head=n_head,
                    attn_dropout_p=attn_dropout_p,
                    resid_dropout_p=resid_dropout_p,
                    ffn_dim_multiplier=ffn_dim_multiplier,
                    ffn_dropout_p=ffn_dropout_p,
                    norm_eps=norm_eps,
                    multiple_of=multiple_of,
                    drop_path=drop_path_rates[i],
                )
                for i in range(n_layer)
            ]
        )
        self.freqs_cis = precompute_freqs_cis(
            seq_len=max_seq_len,
            n_elem=dim // n_head,
            base=rope_base,
            cls_token_num=0,
        )
        self.norm = RMSNorm(dim, eps=norm_eps)

        self.debed = nn.Linear(dim, vocab_size)
        self.unique_decoding = unique_decoding

        self.sinkhorn_decoding = sinkhorn_decoding
        self.sinkhorn_tau = sinkhorn_tau
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_schedule = sinkhorn_schedule
        self.sinkhorn_decay_steps = sinkhorn_decay_steps
        self.sinkhorn_min_tau = sinkhorn_min_tau

        if self.sinkhorn_decoding:
            self.output_query = nn.Parameter(torch.randn(vocab_size, dim) / (dim**0.5))

        self.apply(self.init_weights)

    def sinkhorn_tau_schedule(self, step: int) -> float:
        """Compute the current tau value based on the step and schedule."""
        if self.sinkhorn_schedule == "linear":
            # linearly decay tau from sinkhorn_tau to sinkhorn_min_tau, then keep it constant
            tau = (
                self.sinkhorn_tau
                - (step / self.sinkhorn_decay_steps)
                * (self.sinkhorn_tau - self.sinkhorn_min_tau)
                if step < self.sinkhorn_decay_steps
                else self.sinkhorn_min_tau
            )
        elif self.sinkhorn_schedule == "cosine":
            # cosine decay tau from sinkhorn_tau to sinkhorn_min_tau, then keep it constant
            tau = (
                self.sinkhorn_min_tau
                + 0.5
                * (self.sinkhorn_tau - self.sinkhorn_min_tau)
                * (1 + math.cos((step / self.sinkhorn_decay_steps * math.pi)))
                if step < self.sinkhorn_decay_steps
                else self.sinkhorn_min_tau
            )
        elif self.sinkhorn_schedule == "constant":
            tau = self.sinkhorn_tau
        else:
            raise ValueError(f"Unknown sinkhorn schedule: {self.sinkhorn_schedule}")
        return tau

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            linear_init(module)

    def forward(
        self, tok, mask=None, return_attn=False, return_permutation=False, step=None
    ):
        x = self.embeddings(tok)  # BxNxD floats

        self.freqs_cis = self.freqs_cis.to(
            tok.device
        )  # ensure freqs_cis is on the same device
        freqs_cis = self.freqs_cis[: x.shape[1]]

        if return_attn:
            attn_weights = []
            for block in self.blocks:
                x, attn = block(x, freqs_cis=freqs_cis, mask=mask, return_attn=True)
                attn_weights.append(attn)
            attn_weights = torch.stack(attn_weights, dim=0) # layer x batch x head x N x N
        else:
            for block in self.blocks:
                x = block(x, freqs_cis=freqs_cis, mask=mask)  # BxNxD floats

        x = self.norm(x)  # BxNxD floats

        if self.unique_decoding:
            x = average_embeddings_by_id(
                x, tok, max_vocab_size=self.vocab_size
            )  # pool embeddings by token ID, still BxNxD floats

        if self.sinkhorn_decoding:
            query = self.output_query.unsqueeze(0).expand(x.shape[0], -1, -1)
            x = F.scaled_dot_product_attention(
                query, x, x, attn_mask=mask.squeeze(1) if mask is not None else None
            )  # squeeze to get rid of head dim
            x = self.debed(x)  # BxVxV floats, log permutation matrix over vocab

            if self.training:
                P = gumbel_sinkhorn(
                    x, tau=self.sinkhorn_tau_schedule(step), n_iter=self.sinkhorn_iters
                )  # sample soft permutation matrix
            else:
                P = (
                    torch.stack(
                        [harden_permutation(i) for i in x.cpu().detach().numpy()]
                    )
                    .float()
                    .to(x.device)
                )

            one_hot = F.one_hot(tok, num_classes=self.vocab_size).float()

            x = torch.einsum(
                "bvw,bnv->bnw", P, one_hot
            )  # BxNxV floats, logits over vocab per token
            # x = torch.einsum("bnv,bvw->bnw", one_hot, P.transpose(1, 2))
        else:
            x = self.debed(x)  # BxNxV floats, logits over vocab per token

        outputs = [x]
        if return_attn:
            outputs.append(attn_weights)
        if return_permutation:
            outputs.append(P)
        return tuple(outputs) if len(outputs) > 1 else x
