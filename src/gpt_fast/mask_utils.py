import torch

from torch.nn.attention.flex_attention import (
    create_block_mask,
    BlockMask,
    _mask_mod_signature,
)

create_block_mask_complied = torch.compile(create_block_mask)

def _causal_mask(b, h, q, kv):
    return q >= kv

def left_pad_mask_mod(start_inds, offset) -> _mask_mod_signature:
    def _skip_left_pad_mask(b, h, q, kv):
        return (kv >= start_inds[b]) & ((q + offset) >= kv)

    return _skip_left_pad_mask

def make_base_mask(
    B: int,
    query_len: int,
    kv_len: int,
    device="cuda",
):
    return create_block_mask_complied(
        _causal_mask, B, 1, query_len, kv_len, device=device
    )

def get_gen_submask(
    mask: BlockMask,
    query_idx: int,
) -> BlockMask:
    q_block_size = mask.BLOCK_SIZE[0]
    block_idx = query_idx // q_block_size
    gen_mask = mask[:, :, block_idx:block_idx+1]
    gen_mask.seq_lengths = (1, mask.seq_lengths[1])
    gen_mask.mask_mod = mask.mask_mod
    return gen_mask
