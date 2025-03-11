from gpt_fast.util import maybe_compile

from torch.nn.attention.flex_attention import (
    create_block_mask,
    BlockMask,
    _mask_mod_signature,
)


def _causal_mask(b, h, q, kv):
    return q >= kv


def left_pad_mask_mod(start_inds, offset) -> _mask_mod_signature:
    def _skip_left_pad_mask(b, h, q, kv):
        return (kv >= start_inds[b]) & ((q + offset[0]) >= kv)

    return _skip_left_pad_mask


@maybe_compile
def make_base_mask(
    B: int,
    query_len: int,
    kv_len: int,
    device="cuda",
    **kwargs,
):
    return create_block_mask(
        _causal_mask, B, 1, query_len, kv_len, device=device, **kwargs
    )


def get_gen_submask(
    mask: BlockMask,
    query_idx: int,
) -> BlockMask:
    q_block_size = mask.BLOCK_SIZE[0]
    block_idx = query_idx // q_block_size
    # gen_mask = mask[:, :, block_idx:block_idx+1]
    gen_mask = mask[:, :, block_idx]
    gen_mask.seq_lengths = (1, mask.seq_lengths[1])
    gen_mask.mask_mod = mask.mask_mod
    return gen_mask
