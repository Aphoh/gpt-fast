from gpt_fast.mask_utils import (
    make_base_mask,
    get_prefill_submask,
    get_gen_submask,
)
import torch
from torch.nn.attention.flex_attention import flex_attention


def test_left_pad_mask():
    device = torch.device("cpu")
    B = 4
    BASE_S = 128
    block_size = 4
    start_inds = torch.arange(B) * block_size

    # base gen mask
    mask = make_base_mask(
        start_inds,
        max_seq_length=BASE_S,
        device=device,
        compile=False,
        BLOCK_SIZE=(block_size, block_size),
    )

    print(mask.to_string())
    for i in range(B):
        assert (
            mask.full_kv_num_blocks[i, 0, : i + 1] == 0
        ).all()  # the +1 is just including the 0 index
        assert (mask.full_q_num_blocks[i, 0, :i] == 0).all()


def test_evaluate_mask():
    device = torch.device("cpu")
    B, H, D = 4, 2, 8
    S = 64
    PREFILL_S = 32
    block_size = 4
    start_inds = torch.arange(B) * block_size

    base = make_base_mask(
        start_inds,
        max_seq_length=S,
        device=device,
        compile=False,
        BLOCK_SIZE=(block_size, block_size),
    )

    k, v = (torch.ones(B, H, S, D), torch.ones(B, H, S, D))
    q = torch.ones(B, H, PREFILL_S, D)

    prefill_mask = get_prefill_submask(base, PREFILL_S)
    flex_attention(query=q, key=k, value=v, block_mask=prefill_mask)

    gen_mask = get_gen_submask(base, PREFILL_S)
    q_gen = torch.ones(B, H, 1, D)
    flex_attention(query=q_gen, key=k, value=v, block_mask=gen_mask)
