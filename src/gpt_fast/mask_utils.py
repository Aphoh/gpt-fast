import torch
from gpt_fast.util import maybe_compile

from torch.nn.attention.flex_attention import (
    create_block_mask,
    and_masks,
    BlockMask,
    _mask_mod_signature,
)


def _causal_mask(b, h, q, kv):
    return q >= kv


# TODO add a maximum index to cut off at
def offset_mask_mod(offsets) -> _mask_mod_signature:
    def _offset_mask_mod(b, h, q, kv):
        return (q + offsets[b]) >= kv

    return _offset_mask_mod


def seqlens_mask_mod(seqlens) -> _mask_mod_signature:
    def _end_inds_mask_mod(b, h, q, kv):
        return (kv < seqlens[b]) & (q < seqlens[b])

    return and_masks(_end_inds_mask_mod, _causal_mask)


@maybe_compile
def make_prefill_mask(
    seqlens: torch.Tensor,
    query_len: int,
    max_seqlen: int,
    **kwargs,
):
    (B,) = seqlens.shape
    return create_block_mask(
        seqlens_mask_mod(seqlens),
        B,
        1,
        query_len,
        max_seqlen,
        device=seqlens.device,
        **kwargs,
    )


def get_gen_mask(
    offsets: torch.Tensor,
    max_seqlen: int,
    stop_map: torch.Tensor,
    BLOCK_SIZE: int = 128,
) -> BlockMask:
    (B,) = offsets.shape
    max_blocks = max_seqlen // BLOCK_SIZE
    # we only have one non-full kv block during generation
    kv_num_blocks = (offsets // BLOCK_SIZE) + 1
    kv_num_blocks *= ~stop_map
    # The block which offsets[i] is in is the only block that is not full
    kv_block_inds = torch.arange(
        max_blocks,
        device=offsets.device,
        dtype=torch.int32,
    )[None, :].expand(B, -1)
    # We have 1 less full kv block than the total number of blocks
    full_kv_blocks_num = kv_num_blocks - 1
    full_kv_blocks_num *= ~stop_map
    # These are the same as the kv_block_inds, but without the last block
    full_kv_block_inds = kv_block_inds
    mask_mod = offset_mask_mod(offsets)
    return BlockMask.from_kv_blocks(
        kv_num_blocks=kv_num_blocks[:, None, None],
        kv_indices=kv_block_inds[:, None, None],
        full_kv_num_blocks=full_kv_blocks_num[:, None, None],
        full_kv_indices=full_kv_block_inds[:, None, None],
        BLOCK_SIZE=BLOCK_SIZE,
        mask_mod=mask_mod,
        seq_lengths=(1, max_seqlen),
    )
