from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix


def linear_init(layer):
    nn.init.trunc_normal_(layer.weight, mean=0.0, std=0.02)
    if hasattr(layer, "bias") and layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000, cls_token_num=1):
    freqs = 1.0 / (
        base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)
    )
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)  # (seq_len, head_dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack(
        [freqs_cis.real, freqs_cis.imag], dim=-1
    )  # (cls_token_num+seq_len, head_dim // 2, 2)
    cond_cache = torch.cat(
        [torch.zeros(cls_token_num, n_elem // 2, 2), cache]
    )  # (cls_token_num+seq_len, head_dim // 2, 2)
    return cond_cache


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    # x: (bs, seq_len, n_head, head_dim)
    # freqs_cis (seq_len, head_dim // 2, 2)
    xshaped = x.float().reshape(
        *x.shape[:-1], -1, 2
    )  # (bs, seq_len, n_head, head_dim//2, 2)
    freqs_cis = freqs_cis.view(
        1, xshaped.size(1), 1, xshaped.size(3), 2
    )  # (1, seq_len, 1, head_dim//2, 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        dim=-1,
    )
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


def find_multiple(n: int, k: int):
    if n % k == 0:
        return n
    return n + k - (n % k)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int = 4096,
        ffn_dim_multiplier: Optional[float] = None,
        ffn_dropout_p: float = 0.1,
        multiple_of: int = 256,
    ):
        super().__init__()
        hidden_dim = 4 * dim
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = find_multiple(hidden_dim, multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.ffn_dropout = nn.Dropout(ffn_dropout_p)

    def forward(self, x):
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


def scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
    return_attn=False,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    attn = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )

    if return_attn:
        fake_values = torch.eye(
            query.shape[-2], device=query.device, dtype=query.dtype
        )[*(None,) * (query.ndim - 2), ...]  # (..., seqlen, seqlen)
        attn_weights = F.scaled_dot_product_attention(
            query,
            key,
            fake_values,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )
    else:
        attn_weights = None

    return attn, attn_weights


class Attention(nn.Module):
    def __init__(
        self,
        dim: int = 1024,
        n_head: int = 8,
        attn_dropout_p: float = 0.0,
        resid_dropout_p: float = 0.0,
    ):
        super().__init__()
        assert dim % n_head == 0
        self.dim = dim
        self.head_dim = dim // n_head
        self.n_head = n_head
        total_kv_dim = (3 * self.n_head) * self.head_dim

        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(dim, total_kv_dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        # regularization
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout = nn.Dropout(resid_dropout_p)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor = None,
        mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ):
        bsz, seqlen, _ = x.shape
        kv_size = self.n_head * self.head_dim
        xq, xk, xv = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_head, self.head_dim)

        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)

        xq, keys, values = map(lambda x: x.transpose(1, 2), (xq, xk, xv))

        if mask is not None:
            # match batch size to xq/keys/values
            nrep = bsz // mask.shape[0]
            mask = mask.repeat(nrep, *(1 for _ in range(mask.ndim - 1)))

        output, attn_weights = scaled_dot_product_attention(
            xq,
            keys,
            values,
            attn_mask=mask,
            dropout_p=self.attn_dropout_p if self.training else 0,
            return_attn=return_attn,
        )

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        output = self.resid_dropout(self.wo(output))

        if return_attn:
            return output, attn_weights
        else:
            return output


def log_sinkhorn(x, n_iter):
    for _ in range(n_iter):
        x = x - torch.logsumexp(x, -1, keepdim=True)
        x = x - torch.logsumexp(x, -2, keepdim=True)
    return x.exp()


def harden_permutation(soft_P):
    row, col = linear_sum_assignment(-soft_P)
    permutation_matrix = coo_matrix((np.ones_like(row), (row, col))).toarray()
    return torch.from_numpy(permutation_matrix)


def sample_gumbel(shape, device="cpu", eps=1e-20):
    u = torch.rand(shape, device=device)
    return -torch.log(-torch.log(u + eps) + eps)


def gumbel_sinkhorn(x, tau, n_iter):
    """Sample a permutation matrix from the Gumbel-Sinkhorn distribution
    with parameters given by x and temperature tau.

    Args:
      x: Logarithm of assignment probabilities. In our case this is
        of dimensionality [num_pieces, num_pieces].
      tau: Temperature parameter, the lower the value for tau the more closely
        we follow a categorical sampling.
    """
    gumbel_noise = sample_gumbel(x.shape, device=x.device)
    sampled_perm_mat = log_sinkhorn((x + gumbel_noise) / tau, n_iter)
    return sampled_perm_mat
