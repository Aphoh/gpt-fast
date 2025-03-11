import functools
import itertools
from typing import Any, Dict, Optional, Tuple, Union
import torch
from torch import Tensor
from torch.nn.attention.flex_attention import (
    flex_attention,
    BlockMask,
    _score_mod_signature,
)


def load_model(
    model, checkpoint_path: str, precision: torch.dtype, device: torch.device
):
    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    if model.config.tie_embedding_weights:
        checkpoint["output.weight"] = checkpoint["tok_embeddings.weight"]
    model.load_state_dict(checkpoint, assign=True)
    model = model.to(dtype=precision, device=device)
    return model


def input_pos_from_start_inds(start_inds: torch.Tensor, seq_len: int) -> torch.Tensor:
    (B,) = start_inds.shape
    input_pos_base = torch.arange(seq_len, device=start_inds.device)[None, :].expand(
        B, seq_len
    )
    start_inds_expanded = start_inds[:, None].expand(B, seq_len)
    input_pos = input_pos_base - start_inds_expanded
    return input_pos.clamp(-1)


def _is_power_of_two(n: int) -> bool:
    return (n & (n - 1)) == 0


def _next_power_of_two(n: int) -> int:
    return 1 << (n - 1).bit_length()


def maybe_compile(func=None, **compile_options):
    """
    Decorator that conditionally compiles a PyTorch function.

    Args:
        func: The function to be decorated.
        **compile_options: Options to pass to torch.compile when compiling.

    Returns:
        Decorated function that is optionally compiled.

    Usage:
        @maybe_compile
        def func(x): ...

        @maybe_compile(fullgraph=True, mode='reduce-overhead')
        def func(x): ...
    """

    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, compile=True, **kwargs):
            if compile:
                # Create a compiled version of the function with specified options
                compiled_func = torch.compile(f, **compile_options)
                return compiled_func(*args, **kwargs)
            else:
                # Run the original function
                return f(*args, **kwargs)

        return wrapper

    # Handle both @maybe_compile and @maybe_compile(fullgraph=True) cases
    if func is not None:
        return decorator(func)
    return decorator


# compilation args taken from https://github.com/pytorch/pytorch/issues/142817
# @maybe_compile(fullgraph=True, dynamic=False, mode="max-autotune-no-cudagraphs")
# @torch.compiler.disable(recursive=False)
def flex_attention_maybe_pad(
    query: torch.Tensor,  # (B, Hq, L, E)
    key: Tensor,  # (B, Hkv, S, E)
    value: Tensor,  # (B, Hkv, S, E)
    score_mod: Optional[_score_mod_signature] = None,
    block_mask: Optional[BlockMask] = None,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
    kernel_options: Optional[Dict[str, Any]] = None,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    batch_size, n_q_heads, q_len, head_dims = query.shape
    _, n_kv_heads, kv_len, _ = key.shape

    min_len = min(q_len, kv_len)
    group_size = n_q_heads // n_kv_heads

    if enable_gqa and min_len < 128 and not _is_power_of_two(group_size):
        new_group_size = _next_power_of_two(group_size)
        extra_heads_per_group = new_group_size - group_size

        n_groups = n_q_heads // group_size

        new_n_q_heads = n_q_heads + n_groups * extra_heads_per_group

        # for each group, append extra_heads_per_group of fake heads
        query = (
            torch.concat(
                [
                    query.view(batch_size, n_groups, group_size, q_len, head_dims),
                    query.new_zeros(
                        batch_size, n_groups, extra_heads_per_group, q_len, head_dims
                    ),
                ],
                dim=2,
            )
            .view(batch_size, new_n_q_heads, q_len, head_dims)
            .contiguous()
        )

        result = flex_attention(
            query,
            key,
            value,
            score_mod=score_mod,
            block_mask=block_mask,
            scale=scale,
            enable_gqa=enable_gqa,
            return_lse=return_lse,
            kernel_options=kernel_options,
        )

        attn_out = result if not return_lse else result[0]

        attn_out = attn_out.view(batch_size, n_groups, new_group_size, q_len, head_dims)
        attn_out = attn_out[:, :, :-extra_heads_per_group, :, :]
        attn_out = attn_out.reshape(batch_size, n_q_heads, q_len, head_dims)

        return attn_out if not return_lse else (attn_out, result[1])

    # If no padding is needed, just run flex_attention directly
    return flex_attention(
        query,
        key,
        value,
        score_mod=score_mod,
        block_mask=block_mask,
        scale=scale,
        enable_gqa=enable_gqa,
        return_lse=return_lse,
        kernel_options=kernel_options,
    )


def get_model_size(model):
    model_size = 0
    params = 0
    for name, child in model.named_children():
        if not isinstance(child, torch.nn.Embedding):
            model_size += sum(
                [
                    p.numel() * p.dtype.itemsize
                    for p in itertools.chain(child.parameters(), child.buffers())
                ]
            )
            params += sum(
                [
                    p.numel()
                    for p in itertools.chain(child.parameters(), child.buffers())
                ]
            )
    return model_size, params
