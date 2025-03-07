import torch

from torch.nn.attention.flex_attention import (
    create_block_mask,
    BlockMask,
    _mask_mod_signature,
    and_masks,
)

create_block_mask_complied = torch.compile(create_block_mask)


def _create_block_mask_fn(compile: bool):
    return create_block_mask_complied if compile else create_block_mask


def _causal_mask(b, h, q, kv):
    return q >= kv


def _mask_mod(start_inds) -> _mask_mod_signature:
    def _skip_left_pad_mask(b, h, q, kv):
        return kv >= start_inds[b]

    return and_masks(_skip_left_pad_mask, _causal_mask)


def make_base_mask(
    start_inds: torch.Tensor,
    max_seq_length: int,
    device: torch.device = "cuda",
    compile=True,
    **kwargs,
) -> BlockMask:
    (B,) = start_inds.shape
    return _create_block_mask_fn(compile)(
        _mask_mod(start_inds),
        B=B,
        H=1,
        Q_LEN=max_seq_length,
        KV_LEN=max_seq_length,
        device=device,
        **kwargs,
    )


def get_prefill_submask(
    mask: BlockMask,
    query_len: int,
) -> BlockMask:
    q_block_size = mask.BLOCK_SIZE[0]
    prefill_mask = mask[:, :, : (query_len // q_block_size) + 1]
    prefill_mask.seq_lengths = (query_len, mask.seq_lengths[1])
    return prefill_mask


def get_gen_submask(
    mask: BlockMask,
    query_idx: int,
) -> BlockMask:
    q_block_size = mask.BLOCK_SIZE[0]
    block_idx = query_idx // q_block_size
    gen_mask = mask[:, :, block_idx : block_idx + 1]
    gen_mask.seq_lengths = (1, mask.seq_lengths[1])
    return gen_mask
