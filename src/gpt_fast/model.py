# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from functools import partial
import math
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.attention.flex_attention import BlockMask
from gpt_fast.util import expand_router_probs, flex_attention_maybe_pad, keep_topk
from gpt_fast.mask_utils import offset_mask_mod


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


@dataclass
class RoutableArgs:
    num_experts: int = 64
    expert_rank: int = 16
    top_k: int = 4
    disable_expert_mask: bool = False
    ident_expert_mask: bool = False
    scale: float = 1.0
    prefill_expert: bool = False
    route_each_layer: bool = False
    router_activation: str = "softmax"
    router_act_before_topk: bool = False

    @property
    def prefill_expert_size(self):
        return self.expert_rank * self.top_k if self.prefill_expert else 0

    @property
    def total_expert_rank(self):
        return self.expert_rank * self.num_experts + self.prefill_expert_size


@dataclass
class ModelArgs:
    block_size: int = 2048
    """Maximum sequence length to compute rotary embs for"""
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    head_dim: int = None
    dim: int = 4096
    intermediate_size: int = None
    glu: bool = True
    attn_bias: bool = False
    mlp_bias: bool = False
    n_local_heads: int = -1
    rope_base: float = 10000
    norm_eps: float = 1e-5
    rope_scaling: Optional[dict] = None
    tie_embedding_weights: bool = False
    norm_type: Literal["rmsnorm", "layernorm"] = "rmsnorm"
    act_fn: Literal["silu", "gelu_approx"] = "silu"

    routable_args: Optional[RoutableArgs] = None

    @property
    def is_routed(self) -> bool:
        return self.routable_args is not None

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        if self.head_dim is None:
            self.head_dim = self.dim // self.n_head

    @classmethod
    def from_name(cls, name: str):
        if name in transformer_configs:
            return transformer_configs[name]
        # fuzzy search
        config = [
            config
            for config in transformer_configs
            if config.lower() in str(name).lower()
        ]

        # We may have two or more configs matched (e.g. "7B" and "Mistral-7B"). Find the best config match,
        # take longer name (as it have more symbols matched)
        if len(config) > 1:
            config.sort(key=len, reverse=True)
            assert len(config[0]) != len(config[1]), (
                name
            )  # make sure only one 'best' match

        return transformer_configs[config[0]]

    def make_norm(self, dim=None):
        dim = dim or self.dim
        if self.norm_type == "rmsnorm":
            return RMSNorm(dim, eps=self.norm_eps)
        elif self.norm_type == "layernorm":
            return nn.LayerNorm(dim, eps=self.norm_eps)
        else:
            raise ValueError(f"Unknown norm type: {self.norm_type}")


transformer_configs = {
    "CodeLlama-7b-Python-hf": ModelArgs(
        block_size=16384, vocab_size=32000, n_layer=32, dim=4096, rope_base=1000000
    ),
    "7B": ModelArgs(n_layer=32, n_head=32, dim=4096),
    "13B": ModelArgs(n_layer=40, n_head=40, dim=5120),
    "30B": ModelArgs(n_layer=60, n_head=52, dim=6656),
    "34B": ModelArgs(
        n_layer=48,
        n_head=64,
        dim=8192,
        vocab_size=32000,
        n_local_heads=8,
        intermediate_size=22016,
        rope_base=1000000,
    ),  # CodeLlama-34B-Python-hf
    "70B": ModelArgs(
        n_layer=80, n_head=64, dim=8192, n_local_heads=8, intermediate_size=28672
    ),
    "Mistral-7B": ModelArgs(
        n_layer=32,
        n_head=32,
        n_local_heads=8,
        dim=4096,
        intermediate_size=14336,
        vocab_size=32000,
    ),
    "stories15M": ModelArgs(n_layer=6, n_head=6, dim=288),
    "stories110M": ModelArgs(n_layer=12, n_head=12, dim=768),
    "starcoder2-3b": ModelArgs(
        block_size=16384,
        n_layer=30,
        n_head=24,
        n_local_heads=2,
        dim=3072,
        intermediate_size=12288,
        vocab_size=49152,
        rope_base=500000,
        tie_embedding_weights=True,
        norm_type="layernorm",
        glu=False,
        mlp_bias=True,
        attn_bias=True,
        act_fn="gelu_approx",
    ),
    "llama-3.2-1b": ModelArgs(
        block_size=131072,
        n_layer=16,
        n_head=32,
        n_local_heads=8,
        dim=2048,
        intermediate_size=8192,
        vocab_size=128256,
        rope_base=500000,
        rope_scaling=dict(
            factor=32.0,
            low_freq_factor=1.0,
            high_freq_factor=4.0,
            original_max_position_embeddings=8192,
        ),
        tie_embedding_weights=True,
    ),
    "llama-3-8b": ModelArgs(
        block_size=8192,
        n_layer=32,
        n_head=32,
        n_local_heads=8,
        dim=4096,
        intermediate_size=14336,
        vocab_size=128256,
        rope_base=500000,
    ),
    "llama-3-70b": ModelArgs(
        block_size=8192,
        n_layer=80,
        n_head=64,
        n_local_heads=8,
        dim=8192,
        intermediate_size=28672,
        vocab_size=128256,
        rope_base=500000,
    ),
    "llama-3.1-8b": ModelArgs(
        block_size=131072,
        n_layer=32,
        n_head=32,
        n_local_heads=8,
        dim=4096,
        intermediate_size=14336,
        vocab_size=128256,
        rope_base=500000,
        rope_scaling=dict(
            factor=8.0,
            low_freq_factor=1.0,
            high_freq_factor=4.0,
            original_max_position_embeddings=8192,
        ),
    ),
    "llama-3.1-70b": ModelArgs(
        block_size=131072,
        n_layer=80,
        n_head=64,
        n_local_heads=8,
        dim=8192,
        intermediate_size=28672,
        vocab_size=128256,
        rope_base=500000,
        rope_scaling=dict(
            factor=8.0,
            low_freq_factor=1.0,
            high_freq_factor=4.0,
            original_max_position_embeddings=8192,
        ),
    ),
    "llama-3.1-405b": ModelArgs(
        block_size=131072,
        n_layer=126,
        n_head=128,
        n_local_heads=8,
        dim=16384,
        intermediate_size=53248,
        vocab_size=128256,
        rope_base=500000,
        rope_scaling=dict(
            factor=8.0,
            low_freq_factor=1.0,
            high_freq_factor=4.0,
            original_max_position_embeddings=8192,
        ),
    ),
}


class KVCache(nn.Module):
    def __init__(
        self, max_batch_size, max_seqlen, n_heads, head_dim, dtype=torch.bfloat16
    ):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seqlen, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [B, S], k_val, v_val: [B, H, S, D]
        B = len(input_pos)

        # Get the caches directly
        k_out = self.k_cache
        v_out = self.v_cache

        # Use vectorized indexing - construct proper indices at once
        batch_indices = torch.arange(B, device=input_pos.device)[:, None]

        # Here we index by [B, S]
        # so k_out[batch_indices, :, input_pos] is [B, S, H, D]
        # hence we need to transform k_val: [B, H, S, D] -> [B, S, H, D]
        k_out[batch_indices, :, input_pos] = k_val.transpose(1, 2)
        v_out[batch_indices, :, input_pos] = v_val.transpose(1, 2)

        return k_out, v_out


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layer)
        )
        self.norm = config.make_norm()
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        if config.tie_embedding_weights:
            self.output.weight = self.tok_embeddings.weight

        self.freqs_cis: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seqlen = -1
        self.offset_mask_mod = offset_mask_mod

        self.router: Optional[nn.Linear] = None
        if config.is_routed:
            self.router = nn.Linear(
                config.dim, config.routable_args.num_experts, bias=False
            )
        else:
            self.expert_mask: Optional[Tensor] = None

    def clear_caches(self):
        self.freqs_cis = None
        self.max_batch_size = -1
        self.max_seqlen = -1
        self.expert_mask = None
        for b in self.layers:
            b.attention.kv_cache = None

    def setup_caches(self, max_batch_size, max_seqlen):
        if self.max_seqlen >= max_seqlen and self.max_batch_size >= max_batch_size:
            return
        max_seqlen = find_multiple(max_seqlen, 8)
        self.max_seqlen = max_seqlen
        self.max_batch_size = max_batch_size
        dtype = self.output.weight.dtype
        # For quantized layers, dtype is encoded in scales
        if hasattr(self.output, "scales"):
            dtype = self.output.scales.dtype
        elif hasattr(self.output, "scales_and_zeros"):
            dtype = self.output.scales_and_zeros.dtype
        for b in self.layers:
            b.attention.kv_cache = KVCache(
                max_batch_size,
                max_seqlen,
                self.config.n_local_heads,
                self.config.head_dim,
                dtype,
            )

        self.freqs_cis = precompute_freqs_cis(
            self.config.block_size,
            self.config.head_dim,
            self.config.rope_base,
            dtype,
            self.config.rope_scaling,
        )

        if self.config.is_routed:
            self.register_buffer(
                "expert_mask",
                torch.zeros(
                    (max_batch_size, self.config.routable_args.total_expert_rank),
                    dtype=dtype,
                ),
                persistent=False,
            )

    def forward(
        self,
        mask: BlockMask,
        idx: Tensor,
        input_pos: Tensor,
        is_prefill: bool = False,
    ) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        freqs_cis = self.freqs_cis[input_pos]
        x = self.tok_embeddings(idx)

        if is_prefill and self.config.is_routed:
            rargs = self.config.routable_args
            if rargs.ident_expert_mask:
                self.expert_mask.fill_(1.0)
            else:
                if rargs.prefill_expert:
                    self.expert_mask[:, : rargs.prefill_expert_size] = 1.0
                    self.expert_mask[:, rargs.prefill_expert_size :] = 0.0
                else:
                    self.expert_mask[:] = 0.0

        for i, layer in enumerate(self.layers):
            x = layer(x, freqs_cis, mask, input_pos, self.expert_mask)
        x = self.norm(x)

        if is_prefill and self.config.is_routed:
            if rargs.ident_expert_mask:
                self.expert_mask.fill_(1.0)
            else:
                rargs = self.config.routable_args
                router_input = x[:, -1]
                router_logits = self.router(router_input)
                if rargs.router_act_before_topk:
                    router_probs = F.softmax(router_logits, dim=-1)
                    router_probs = keep_topk(router_probs, rargs.top_k)
                else:
                    router_logits = keep_topk(router_logits, rargs.top_k)
                    router_probs = F.softmax(router_logits, dim=-1)
                mask_probs = expand_router_probs(router_probs, rargs.expert_rank)
                if rargs.prefill_expert:
                    self.expert_mask[:, : rargs.prefill_expert_size] = 0.0
                    self.expert_mask[:, rargs.prefill_expert_size :] = mask_probs
                else:
                    self.expert_mask[:] = mask_probs

        logits = self.output(x)
        return logits

    @classmethod
    def from_name(cls, name: str):
        return cls(ModelArgs.from_name(name))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = config.make_norm()
        self.attention_norm = config.make_norm()
        self.routed_experts: Optional[RoutableExperts] = None
        if config.is_routed:
            self.routed_experts = RoutableExperts(config)

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        mask: BlockMask,
        input_pos: Tensor,
        expert_mask: Optional[Tensor] = None,
    ) -> Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)
        ffn_input = self.ffn_norm(h)
        out = h + self.feed_forward(ffn_input)
        if self.routed_experts and expert_mask is not None:
            out += self.routed_experts(ffn_input, expert_mask)
        return out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=config.attn_bias)
        self.wo = nn.Linear(config.dim, config.dim, bias=config.attn_bias)
        self.kv_cache: Optional[KVCache] = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        mask: BlockMask,
        input_pos: torch.Tensor,
    ) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        y = flex_attention_maybe_pad(
            q, k, v, block_mask=mask, enable_gqa=(self.n_head != self.n_local_heads)
        )

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        y = self.wo(y)
        return y


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=config.mlp_bias)
        if config.glu:
            self.w3 = nn.Linear(
                config.dim, config.intermediate_size, bias=config.mlp_bias
            )
        else:
            self.w3 = None
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=config.mlp_bias)
        self.act = (
            F.silu if config.act_fn == "silu" else partial(F.gelu, approximate="tanh")
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.act(self.w1(x))
        if self.w3:
            out *= self.w3(x)
        return self.w2(out)


class RoutableExperts(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        rargs = config.routable_args
        self.w1 = nn.Linear(config.dim, rargs.total_expert_rank, bias=False)
        if config.glu:
            self.w3 = nn.Linear(config.dim, rargs.total_expert_rank, bias=False)
        else:
            self.w3 = None
        self.w2 = nn.Linear(rargs.total_expert_rank, config.dim, bias=False)
        self.act = (
            F.silu if config.act_fn == "silu" else partial(F.gelu, approximate="tanh")
        )

    def forward(self, x: Tensor, expert_mask: torch.Tensor) -> Tensor:
        out = self.act(self.w1(x))
        if self.w3:
            out *= self.w3(x)
        out *= expert_mask[:, None]
        return self.w2(out)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def apply_rope_scaling(freqs: torch.Tensor, rope_scaling: Optional[dict] = None):
    factor = rope_scaling["factor"]
    low_freq_factor = rope_scaling["low_freq_factor"]
    high_freq_factor = rope_scaling["high_freq_factor"]
    old_context_len = rope_scaling["original_max_position_embeddings"]

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs_cis(
    seq_len: int,
    n_elem: int,
    base: int = 10000,
    dtype: torch.dtype = torch.bfloat16,
    rope_scaling: Optional[dict] = None,
) -> Tensor:
    freqs = 1.0 / (
        base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)
    )
    if rope_scaling is not None:
        freqs = apply_rope_scaling(freqs, rope_scaling)
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    # x: [B, S, H, D]
    # freqs_cis: [B, S, H//2, 2]
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)
