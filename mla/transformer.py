from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class AttentionType(Enum):
    MULTI_HEAD_ATTENTION = "mha"
    MULTI_LATENT_ATTENTION_NAIVE = "mla_naive"
    MULTI_LATENT_ATTENTION_OPTIMIZED = "mla_optimized"


@dataclass
class ModelArgs:
    dim: int = 1024
    vocab_size: int = 32768
    n_layers: int = 16
    n_heads: int = 8
    n_kv_heads: int = 8
    padding_idx: int = 0
    norm_eps: float = 1e-8
    intermediate_size: int = 4096
    rope_theta: float = 10000
    max_position_embeddings: int = 8192
    attention_type: AttentionType = AttentionType.MULTI_HEAD_ATTENTION

    # MLA specific
    q_compressed_dim: int = 128
    q_nope_head_dim: int = 96
    q_rope_head_dim: int = 32
    kv_compressed_dim: int = 128
    k_nope_head_dim: int = 64
    k_rope_head_dim: int = 64
    v_head_dim: int = 256

    def __post_init__(self):
        if self.dim % self.n_heads != 0:
            raise ValueError(
                f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})"
            )

        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
            )

        if (
            self.attention_type == AttentionType.MULTI_LATENT_ATTENTION_NAIVE
            or self.attention_type == AttentionType.MULTI_LATENT_ATTENTION_OPTIMIZED
        ):
            assert (
                self.q_nope_head_dim + self.q_rope_head_dim
                == self.k_nope_head_dim + self.k_rope_head_dim
            ), (
                (self.q_nope_head_dim + self.q_rope_head_dim),
                (self.k_nope_head_dim + self.k_rope_head_dim),
            )

    @property
    def gqa_factor(self) -> int:
        return self.n_heads // self.n_kv_heads

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads


@dataclass
class KVCache:
    @abstractmethod
    def num_tokens(self) -> int: ...


class MultiHeadAttentionKVCache(KVCache):
    def __init__(self):
        self.k = []
        self.v = []

    @property
    def num_tokens(self) -> int:
        if len(self.k) == 0:
            return 0
        return self.k[0].shape[-2]

    def add_and_retrieve(
        self, layer_idx: int, k: Tensor, v: Tensor
    ) -> tuple[Tensor, Tensor]:
        # Shape: [Batch, Num_Heads, Seq_Len, Head_Dim]
        if len(self.k) <= layer_idx:
            self.k.append(k)
            self.v.append(v)
        else:
            self.k[layer_idx] = torch.cat((self.k[layer_idx], k), dim=-2)
            self.v[layer_idx] = torch.cat((self.v[layer_idx], v), dim=-2)
        return self.k[layer_idx], self.v[layer_idx]


class MultiLatentAttentionKVCache(KVCache):
    def __init__(self):
        self.latent_kv = []
        self.k_rope = []

    @property
    def num_tokens(self) -> int:
        if len(self.latent_kv) == 0:
            return 0
        return self.latent_kv[0].shape[-2]

    def add_and_retrieve(
        self, layer_idx: int, c_kv: Tensor, k_rope: Tensor
    ) -> tuple[Tensor, Tensor]:
        # c_kv shape: [Batch, Seq_Len, KV_Compressed_Dim]
        # k_rope shape: [Batch, Seq_Len, 1 * K_Rope_Head_Dim]

        if len(self.latent_kv) <= layer_idx:
            self.latent_kv.append(c_kv)
            self.k_rope.append(k_rope)
        else:
            self.latent_kv[layer_idx] = torch.cat(
                (self.latent_kv[layer_idx], c_kv), dim=-2
            )
            self.k_rope[layer_idx] = torch.cat((self.k_rope[layer_idx], k_rope), dim=-2)
        return self.latent_kv[layer_idx], self.k_rope[layer_idx]


def repeat_kv_heads(x: Tensor, n_rep: int) -> Tensor:
    if x.dim() != 4:
        raise ValueError(f"Expected 4D tensor, got {x.dim()}D")

    return torch.repeat_interleave(x, n_rep, dim=1)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, args: ModelArgs, head_dim: int):
        super().__init__()
        if head_dim % 2:
            raise ValueError("head_dim must be even")

        self.head_dim = head_dim
        self.max_position_embeddings = args.max_position_embeddings

        inv_freq = 1.0 / (
            args.rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        positions = torch.arange(args.max_position_embeddings, dtype=torch.float32)
        theta = torch.outer(positions, inv_freq)

        self.register_buffer(
            "cos_cached", theta.cos(), persistent=False
        )  # [Max_Position_Embeddings, Head_Dim // 2]
        self.register_buffer(
            "sin_cached", theta.sin(), persistent=False
        )  # [Max_Position_Embeddings, Head_Dim // 2]

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or position_ids.dim() != 2:
            raise ValueError("x must be 4D and position_ids 2D")
        if position_ids.max() >= self.max_position_embeddings:
            raise ValueError(f"position_ids must be < {self.max_position_embeddings}")

        cos = self.cos_cached[position_ids, :]  # [Batch, Seq_Len, Head_Dim // 2]
        sin = self.sin_cached[position_ids, :]  # [Batch, Seq_Len, Head_Dim // 2]
        cos = cos.unsqueeze(1)  # [Batch, 1, Seq_Len, Head_Dim // 2]
        sin = sin.unsqueeze(1)  # [Batch, 1, Seq_Len, Head_Dim // 2]

        cos = cos.to(dtype=x.dtype, device=x.device)
        sin = sin.to(dtype=x.dtype, device=x.device)

        x_even = x[..., ::2]  # [Batch, Heads, Seq_Len, Head_Dim // 2]
        x_odd = x[..., 1::2]  # [Batch, Heads, Seq_Len, Head_Dim // 2]

        rot_even = x_even * cos - x_odd * sin
        rot_odd = x_even * sin + x_odd * cos

        x_rotated = torch.stack((rot_even, rot_odd), dim=-1)
        x_rotated = x_rotated.flatten(-2)

        return x_rotated


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return (
            self.weight
            * x
            * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        )


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.up_projection = nn.Linear(args.dim, args.intermediate_size)
        self.gate_projection = nn.Linear(args.dim, args.intermediate_size)
        self.down_projection = nn.Linear(args.intermediate_size, args.dim)
        self.act_fn = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.down_projection(
            self.act_fn(self.gate_projection(x)) * self.up_projection(x)
        )


class MultiLatentAttentionOptimized(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.args = args
        self.layer_idx = layer_idx

        self.w_dq = nn.Linear(args.dim, args.q_compressed_dim)
        self.q_norm = RMSNorm(args.q_compressed_dim, args.norm_eps)
        # We can combine two projections into one
        self.w_uq_qr = nn.Linear(
            args.q_compressed_dim,
            args.n_heads * (args.q_nope_head_dim + args.q_rope_head_dim),
        )

        self.rope_q = RotaryPositionalEmbedding(args, args.q_rope_head_dim)
        self.rope_k = RotaryPositionalEmbedding(args, args.k_rope_head_dim)

        self.w_dkv_kr = nn.Linear(
            args.dim, (args.kv_compressed_dim + args.k_rope_head_dim)
        )
        self.kv_norm = RMSNorm(args.kv_compressed_dim, args.norm_eps)
        self.w_uk_uv = nn.Linear(
            args.kv_compressed_dim,
            args.n_kv_heads * (args.k_nope_head_dim + args.v_head_dim),
        )
        self.w_o = nn.Linear(args.n_heads * args.v_head_dim, args.dim)

    def forward(
        self,
        x: Tensor,
        position_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tensor:
        batch_size, q_seq_len, _ = x.shape

        compressed_q = self.q_norm(
            self.w_dq(x)
        )  # [Batch_Size, Seq_Len, Q_Compressed_Dim]

        compressed_kv_k_rope = self.w_dkv_kr(
            x
        )  # [Batch_Size, Seq_Len, KV_Compressed_Dim + K_Rope_Head_Dim]

        compressed_kv, k_rope = compressed_kv_k_rope.split(
            [self.args.kv_compressed_dim, self.args.k_rope_head_dim], dim=-1
        )

        compressed_kv = self.kv_norm(compressed_kv)

        # [Batch_Size, Seq_Len, 1 * K_Rope_Head_Dim] -> [Batch_Size, 1, Seq_Len, K_Rope_Head_Dim]
        k_rope = k_rope.view(
            batch_size, q_seq_len, 1, self.args.k_rope_head_dim
        ).transpose(1, 2)

        k_rope = self.rope_k(k_rope, position_ids)

        if kv_cache is not None:
            compressed_kv, k_rope = kv_cache.add_and_retrieve(
                self.layer_idx, c_kv=compressed_kv, k_rope=k_rope
            )

        k_seq_len = compressed_kv.shape[-2]

        q_nope_q_rope = self.w_uq_qr(
            compressed_q
        )  # [Batch_Size, Seq_Len, Num_Heads * (Q_Nope_Head_Dim + Q_Rope_Head_Dim)]

        q_nope, q_rope = q_nope_q_rope.split(
            [
                self.args.n_heads * self.args.q_nope_head_dim,
                self.args.n_heads * self.args.q_rope_head_dim,
            ],
            dim=-1,
        )

        # # [Batch_Size, Seq_Len, Num_Heads * Q_Nope_Head_Dim] --> [Batch_Size, Num_Heads, Seq_Len, Q_Nope_Head_Dim]
        q_nope = q_nope.view(
            batch_size, q_seq_len, self.args.n_heads, self.args.q_nope_head_dim
        ).transpose(1, 2)

        # # [Batch_Size, Seq_Len, Num_Heads * Q_Rope_Head_Dim] --> [Batch_Size, Num_Heads, Seq_Len, Q_Rope_Head_Dim]
        q_rope = q_rope.view(
            batch_size, q_seq_len, self.args.n_heads, self.args.q_rope_head_dim
        ).transpose(1, 2)

        # Apply RoPE to q
        q_rope = self.rope_q(q_rope, position_ids)

        # Concatenate NoPE and RoPE
        # [Batch_Size, Num_Heads, Seq_Len, Q_Nope_Head_Dim + Q_Rope_Head_Dim]
        query_states = torch.cat((q_nope, q_rope), dim=-1)

        assert query_states.shape == (
            batch_size,
            self.args.n_heads,
            q_seq_len,
            (self.args.q_nope_head_dim + self.args.q_rope_head_dim),
        )

        k_nope_v_states = self.w_uk_uv(
            compressed_kv
        )  # [Batch_Size, Seq_Len, Num_KV_Heads * (K_Nope_Head_Dim + V_Head_Dim)]

        k_nope, v_states = k_nope_v_states.split(
            [
                self.args.n_kv_heads * self.args.k_nope_head_dim,
                self.args.n_kv_heads * self.args.v_head_dim,
            ],
            dim=-1,
        )

        # [Batch_Size, Seq_Len, Num_KV_Heads * K_Nope_Head_Dim] -> [Batch_Size, Num_KV_Heads, Seq_Len, K_Nope_Head_Dim]
        k_nope = k_nope.view(
            batch_size, k_seq_len, self.args.n_kv_heads, self.args.k_nope_head_dim
        ).transpose(1, 2)

        # We need to repeat the k_rope number of heads to match that of k_nope
        assert k_rope.shape[1] == 1, "k_rope must have 1 head"
        k_rope = repeat_kv_heads(k_rope, self.args.n_kv_heads)

        # Concatenate NoPE and RoPE
        # [Batch_Size, Num_KV_Heads, Seq_Len, K_Nope_Head_Dim + K_Rope_Head_Dim]
        k_states = torch.cat((k_nope, k_rope), dim=-1)

        # Sanity check
        assert k_states.shape == (
            batch_size,
            self.args.n_kv_heads,
            k_seq_len,
            (self.args.k_nope_head_dim + self.args.k_rope_head_dim),
        )

        # Repeat heads to match the number of heads in the Q
        k_states = repeat_kv_heads(k_states, self.args.gqa_factor)

        v_states = v_states.view(
            batch_size, k_seq_len, self.args.n_kv_heads, self.args.v_head_dim
        ).transpose(1, 2)

        # Repeat heads to match the number of heads in the Q
        v_states = repeat_kv_heads(v_states, self.args.gqa_factor)

        # Compute the attention weights
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query=query_states,
            key=k_states,
            value=v_states,
            attn_mask=attention_mask,
        )

        # Move the heads to the last dimension
        # [Batch_Size, Num_Heads, Seq_Len, V_Head_Dim] -> [Batch_Size, Seq_Len, Num_Heads * V_Head_Dim]
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, q_seq_len, self.args.n_heads * self.args.v_head_dim
        )

        # [Batch_Size, Seq_Len, Dim]
        return self.w_o(attn_output)


class MultiLatentAttentionNaive(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.args = args
        self.layer_idx = layer_idx

        self.w_dq = nn.Linear(args.dim, args.q_compressed_dim)
        self.q_norm = RMSNorm(args.q_compressed_dim, args.norm_eps)
        self.w_uq = nn.Linear(
            args.q_compressed_dim, args.n_heads * args.q_nope_head_dim
        )
        self.w_qr = nn.Linear(
            args.q_compressed_dim, args.n_heads * args.q_rope_head_dim
        )

        self.rope_q = RotaryPositionalEmbedding(args, args.q_rope_head_dim)
        self.rope_k = RotaryPositionalEmbedding(args, args.k_rope_head_dim)

        self.w_dkv = nn.Linear(args.dim, args.kv_compressed_dim)
        self.kv_norm = RMSNorm(args.kv_compressed_dim, args.norm_eps)
        self.w_uk = nn.Linear(
            args.kv_compressed_dim, args.n_kv_heads * args.k_nope_head_dim
        )

        self.w_kr = nn.Linear(args.dim, 1 * args.k_rope_head_dim)

        self.w_uv = nn.Linear(args.kv_compressed_dim, args.n_kv_heads * args.v_head_dim)

        self.w_o = nn.Linear(args.n_heads * args.v_head_dim, args.dim)

    def forward(
        self,
        x: Tensor,
        position_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tensor:
        batch_size, q_seq_len, _ = x.shape

        compressed_q = self.q_norm(
            self.w_dq(x)
        )  # [Batch_Size, Seq_Len, Q_Compressed_Dim]

        compressed_kv = self.kv_norm(
            self.w_dkv(x)
        )  # [Batch_Size, Seq_Len, KV_Compressed_Dim]

        k_rope = self.w_kr(x)  # [Batch_Size, Seq_Len, 1 * K_Rope_Head_Dim]

        # [Batch_Size, Seq_Len, 1 * K_Rope_Head_Dim] -> [Batch_Size, 1, Seq_Len, K_Rope_Head_Dim]
        k_rope = k_rope.view(
            batch_size, q_seq_len, 1, self.args.k_rope_head_dim
        ).transpose(1, 2)

        k_rope = self.rope_k(k_rope, position_ids)

        if kv_cache is not None:
            compressed_kv, k_rope = kv_cache.add_and_retrieve(
                self.layer_idx, c_kv=compressed_kv, k_rope=k_rope
            )

        k_seq_len = compressed_kv.shape[-2]

        q_nope = self.w_uq(
            compressed_q
        )  # [Batch_Size, Seq_Len, Num_Heads * Q_Nope_Head_Dim]
        q_rope = self.w_qr(
            compressed_q
        )  # [Batch_Size, Seq_Len, Num_Heads * Q_Rope_Head_Dim]

        # # [Batch_Size, Seq_Len, Num_Heads * Q_Nope_Head_Dim] --> [Batch_Size, Num_Heads, Seq_Len, Q_Nope_Head_Dim]
        q_nope = q_nope.view(
            batch_size, q_seq_len, self.args.n_heads, self.args.q_nope_head_dim
        ).transpose(1, 2)

        # # [Batch_Size, Seq_Len, Num_Heads * Q_Rope_Head_Dim] --> [Batch_Size, Num_Heads, Seq_Len, Q_Rope_Head_Dim]
        q_rope = q_rope.view(
            batch_size, q_seq_len, self.args.n_heads, self.args.q_rope_head_dim
        ).transpose(1, 2)

        # Apply RoPE to q
        q_rope = self.rope_q(q_rope, position_ids)

        # Concatenate NoPE and RoPE
        # [Batch_Size, Num_Heads, Seq_Len, Q_Nope_Head_Dim + Q_Rope_Head_Dim]
        query_states = torch.cat((q_nope, q_rope), dim=-1)

        assert query_states.shape == (
            batch_size,
            self.args.n_heads,
            q_seq_len,
            (self.args.q_nope_head_dim + self.args.q_rope_head_dim),
        )

        k_nope = self.w_uk(
            compressed_kv
        )  # [Batch_Size, Seq_Len, Num_KV_Heads * K_Nope_Head_Dim]

        # [Batch_Size, Seq_Len, Num_KV_Heads * K_Nope_Head_Dim] -> [Batch_Size, Num_KV_Heads, Seq_Len, K_Nope_Head_Dim]
        k_nope = k_nope.view(
            batch_size, k_seq_len, self.args.n_kv_heads, self.args.k_nope_head_dim
        ).transpose(1, 2)

        # We need to repeat the k_rope number of heads to match that of k_nope
        assert k_rope.shape[1] == 1, "k_rope must have 1 head"
        k_rope = repeat_kv_heads(k_rope, self.args.n_kv_heads)

        # Concatenate NoPE and RoPE
        # [Batch_Size, Num_KV_Heads, Seq_Len, K_Nope_Head_Dim + K_Rope_Head_Dim]
        k_states = torch.cat((k_nope, k_rope), dim=-1)

        # Sanity check
        assert k_states.shape == (
            batch_size,
            self.args.n_kv_heads,
            k_seq_len,
            (self.args.k_nope_head_dim + self.args.k_rope_head_dim),
        )

        # Repeat heads to match the number of heads in the Q
        k_states = repeat_kv_heads(k_states, self.args.gqa_factor)

        v_states = self.w_uv(
            compressed_kv
        )  # [Batch_Size, Seq_Len, Num_KV_Heads * V_Head_Dim]

        v_states = v_states.view(
            batch_size, k_seq_len, self.args.n_kv_heads, self.args.v_head_dim
        ).transpose(1, 2)

        # Repeat heads to match the number of heads in the Q
        v_states = repeat_kv_heads(v_states, self.args.gqa_factor)

        # Compute the attention weights
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query=query_states,
            key=k_states,
            value=v_states,
            attn_mask=attention_mask,
        )

        # Move the heads to the last dimension
        # [Batch_Size, Num_Heads, Seq_Len, V_Head_Dim] -> [Batch_Size, Seq_Len, Num_Heads * V_Head_Dim]
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, q_seq_len, self.args.n_heads * self.args.v_head_dim
        )

        # [Batch_Size, Seq_Len, Dim]
        return self.w_o(attn_output)

    @staticmethod
    def convert_naive_to_optimized(
        naive_attn: "MultiLatentAttentionNaive",
    ) -> "MultiLatentAttentionOptimized":
        args = naive_attn.args
        opt_attn = MultiLatentAttentionOptimized(args, layer_idx=naive_attn.layer_idx)
        # Copy d_wq
        opt_attn.w_dq.weight = nn.Parameter(naive_attn.w_dq.weight.clone())
        if opt_attn.w_dq.bias is not None:
            opt_attn.w_dq.bias = nn.Parameter(naive_attn.w_dq.bias.clone())
        # Copy q_norm
        opt_attn.q_norm.weight = nn.Parameter(naive_attn.q_norm.weight.clone())
        # Copy w_uq and w_qr
        opt_attn.w_uq_qr.weight = nn.Parameter(
            torch.cat(
                [naive_attn.w_uq.weight.clone(), naive_attn.w_qr.weight.clone()], dim=0
            )
        )
        if opt_attn.w_uq_qr.bias is not None:
            opt_attn.w_uq_qr.bias = nn.Parameter(
                torch.cat(
                    [naive_attn.w_uq.bias.clone(), naive_attn.w_qr.bias.clone()], dim=0
                )
            )
        # Copy w_dkv and w_kr
        opt_attn.w_dkv_kr.weight = nn.Parameter(
            torch.cat(
                [naive_attn.w_dkv.weight.clone(), naive_attn.w_kr.weight.clone()], dim=0
            )
        )
        if opt_attn.w_dkv_kr.bias is not None:
            opt_attn.w_dkv_kr.bias = nn.Parameter(
                torch.cat(
                    [naive_attn.w_dkv.bias.clone(), naive_attn.w_kr.bias.clone()], dim=0
                )
            )
        # Copy kv_norm
        opt_attn.kv_norm.weight = nn.Parameter(naive_attn.kv_norm.weight.clone())
        # Copy w_uk and w_uv
        opt_attn.w_uk_uv.weight = nn.Parameter(
            torch.cat(
                [naive_attn.w_uk.weight.clone(), naive_attn.w_uv.weight.clone()], dim=0
            )
        )
        if opt_attn.w_uk_uv.bias is not None:
            opt_attn.w_uk_uv.bias = nn.Parameter(
                torch.cat(
                    [naive_attn.w_uk.bias.clone(), naive_attn.w_uv.bias.clone()], dim=0
                )
            )
        # Copy w_o
        opt_attn.w_o.weight = nn.Parameter(naive_attn.w_o.weight.clone())
        if opt_attn.w_o.bias is not None:
            opt_attn.w_o.bias = nn.Parameter(naive_attn.w_o.bias.clone())
        return opt_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.args = args
        self.layer_idx = layer_idx

        self.w_q = nn.Linear(args.dim, args.head_dim * args.n_heads)
        self.w_k = nn.Linear(args.dim, args.head_dim * args.n_kv_heads)
        self.w_v = nn.Linear(args.dim, args.head_dim * args.n_kv_heads)

        self.w_o = nn.Linear(args.head_dim * args.n_heads, args.dim)

        self.rope = RotaryPositionalEmbedding(args, args.head_dim)

    def forward(
        self,
        x: Tensor,
        position_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tensor:
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        batch_size, seq_len, _ = q.shape

        q = q.view(
            batch_size, seq_len, self.args.n_heads, self.args.head_dim
        ).transpose(1, 2)
        k = k.view(
            batch_size, seq_len, self.args.n_kv_heads, self.args.head_dim
        ).transpose(1, 2)
        v = v.view(
            batch_size, seq_len, self.args.n_kv_heads, self.args.head_dim
        ).transpose(1, 2)

        # Apply RoPE rotation to the query and key
        q = self.rope(q, position_ids)
        k = self.rope(k, position_ids)

        # Add to the key and value cache
        if kv_cache is not None:
            k, v = kv_cache.add_and_retrieve(self.layer_idx, k, v)

        # Repeat the key and value for each head
        k = repeat_kv_heads(k, self.args.gqa_factor)
        v = repeat_kv_heads(v, self.args.gqa_factor)

        # Compute the attention weights
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask
        )

        # Move the heads to the last dimension
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, seq_len, self.args.n_heads * self.args.head_dim
        )

        return self.w_o(attn_output)


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.args = args

        self.norm_attention = RMSNorm(args.dim, args.norm_eps)
        self.norm_mlp = RMSNorm(args.dim, args.norm_eps)

        if args.attention_type == AttentionType.MULTI_HEAD_ATTENTION:
            self.attention = MultiHeadAttention(args, layer_idx)
        elif args.attention_type == AttentionType.MULTI_LATENT_ATTENTION_NAIVE:
            self.attention = MultiLatentAttentionNaive(args, layer_idx)
        elif args.attention_type == AttentionType.MULTI_LATENT_ATTENTION_OPTIMIZED:
            self.attention = MultiLatentAttentionOptimized(args, layer_idx)
        else:
            raise ValueError(f"Unsupported attention type: {args.attention_type}")
        self.mlp = MLP(args)

    def forward(
        self,
        x: Tensor,
        position_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tensor:
        x = x + self.attention(
            x=self.norm_attention(x),
            position_ids=position_ids,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )
        x = x + self.mlp(self.norm_mlp(x))

        return x


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.embeddings = nn.Embedding(
            args.vocab_size, args.dim, padding_idx=args.padding_idx
        )

        self.norm = RMSNorm(args.dim, args.norm_eps)

        self.layers = nn.ModuleList(
            [TransformerBlock(args, layer_idx) for layer_idx in range(args.n_layers)]
        )

        self.lm_head = nn.Linear(args.dim, args.vocab_size)

    def convert_mask_from_2d_to_4d(
        self,
        hidden_states: Tensor,  # [Batch, Seq_Len, Dim]
        attention_mask_2d: Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> Tensor:
        batch_size, q_size, _ = hidden_states.shape

        past_kv_len = kv_cache.num_tokens if kv_cache is not None else 0
        kv_size = past_kv_len + q_size

        if (
            attention_mask_2d.dim() != 2
            or attention_mask_2d.shape[0] != batch_size
            or attention_mask_2d.shape[1] != kv_size
        ):
            raise ValueError(
                f"attention_mask_2d has shape {attention_mask_2d.shape}, "
                f"but expected shape [batch_size, kv_size] = [{batch_size}, {kv_size}]"
            )

        mask_dtype = hidden_states.dtype

        attention_mask_4d = torch.ones(
            batch_size,
            1,
            q_size,
            kv_size,
            device=hidden_states.device,
            dtype=mask_dtype,
        )

        attention_mask_4d = torch.tril(attention_mask_4d, diagonal=past_kv_len)

        expanded_padding_mask = attention_mask_2d.unsqueeze(1).unsqueeze(2)

        expanded_padding_mask = expanded_padding_mask.to(
            device=attention_mask_4d.device, dtype=attention_mask_4d.dtype
        )

        attention_mask_4d = attention_mask_4d * expanded_padding_mask

        # Convert the mask into an additive mask
        negative_infinity = torch.finfo(attention_mask_4d.dtype).min
        attention_mask_4d = torch.masked_fill(
            attention_mask_4d, attention_mask_4d == 0, negative_infinity
        )
        attention_mask_4d = torch.masked_fill(
            attention_mask_4d, attention_mask_4d == 1, 0
        )

        return attention_mask_4d

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tensor:
        hidden_states = self.embeddings(input_ids)

        attention_mask_4d = self.convert_mask_from_2d_to_4d(
            hidden_states, attention_mask, kv_cache
        )

        for layer in self.layers:
            hidden_states = layer(
                x=hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask_4d,
                kv_cache=kv_cache,
            )

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits
