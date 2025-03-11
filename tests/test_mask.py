from gpt_fast.mask_utils import left_pad_mask_mod, make_base_mask, get_gen_submask
import torch
from torch.nn.attention.flex_attention import flex_attention


def test_left_pad_mask():
    device = torch.device("cpu")
    B = 4
    BASE_S = 16
    block_size = 4
    start_inds = torch.arange(B) * block_size

    all_base_mask = make_base_mask(B, BASE_S, BASE_S, device=device, compile=False)
    all_base_mask.mask_mod = left_pad_mask_mod(start_inds, torch.tensor([0]))

    gen_base = make_base_mask(B, BASE_S, BASE_S, device=device, compile=False)

    with torch.no_grad():
        qkv = torch.randn(B, 1, BASE_S, BASE_S, device=device)
        all_out = flex_attention(qkv, qkv, qkv, block_mask=all_base_mask)

        for query_idx in range(BASE_S):
            gen_submask = get_gen_submask(gen_base, query_idx)
            gen_submask.mask_mod = left_pad_mask_mod(
                start_inds, torch.tensor([query_idx])
            )
            gen_out = flex_attention(
                qkv[:, :, query_idx : query_idx + 1], qkv, qkv, block_mask=gen_submask
            )
            assert torch.allclose(all_out[:, :, query_idx], gen_out[:, :, 0])


def test_left_pad_works():
    device = torch.device("cpu")
    qkv = torch.randn(2, 1, 32, 4, device=device)
    # same sequence, but batch 1 has a single left pad
    qkv[1, :, 1:] = qkv[0, :, :-1]
    qkv[1, :, 0] = 0
    start_inds = torch.tensor([0, 1], device=device)

    mask = make_base_mask(start_inds.shape[0], 32, 32, device=device, compile=False)
    mask.mask_mod = left_pad_mask_mod(start_inds, [0])
    output = flex_attention(qkv, qkv, qkv, block_mask=mask)

    assert torch.allclose(output[0, :, :-1], output[1, :, 1:])
