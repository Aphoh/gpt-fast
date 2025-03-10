from gpt_fast.mask_utils import (
    left_pad_mask_mod,
    make_base_mask,
    make_prefill_mask,
)
import torch
from torch.nn.attention.flex_attention import flex_attention


def test_left_pad_mask():
    device = torch.device("cpu")
    B = 4
    BASE_S = 128
    block_size = 4
    start_inds = torch.arange(B) * block_size

    mask = make_prefill_mask(
        start_inds, BASE_S, BASE_S, device=device, BLOCK_SIZE=(block_size, block_size)
    )

    print(mask.to_string())
    for i in range(B):
        assert (
            mask.full_kv_num_blocks[i, 0, : i + 1] == 0
        ).all()  # the +1 is just including the 0 index
        assert (mask.full_q_num_blocks[i, 0, :i] == 0).all()

    # base gen mask
    gen_mask = make_base_mask(
        B,
        BASE_S,
        BASE_S,
        device=device,
        BLOCK_SIZE=(block_size, block_size),
    )
    gen_mask.mask_mod = left_pad_mask_mod(start_inds, [0])
    print(gen_mask.to_string())

    with torch.no_grad():
        qkv = torch.randn(B, 1, BASE_S, BASE_S, device=device)
        gen_out = flex_attention(qkv, qkv, qkv, block_mask=gen_mask)
        prefill_out = flex_attention(qkv, qkv, qkv, block_mask=mask)
        assert torch.allclose(gen_out, prefill_out)


def test_left_pad_works():
    device = torch.device("cpu")
    qkv = torch.randn(2, 1, 32, 4, device=device)
    # same sequence, but batch 1 has a single left pad
    qkv[1, :, 1:] = qkv[0, :, :-1]
    qkv[1, :, 0] = 0
    start_inds = torch.tensor([0, 1], device=device)

    mask = make_prefill_mask(start_inds, 32, 32, device=device)
    output = flex_attention(qkv, qkv, qkv, block_mask=mask)

    assert torch.allclose(output[0, :, :-1], output[1, :, 1:])
