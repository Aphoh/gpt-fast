from gpt_fast.mask_utils import offset_mask_mod, make_prefill_mask, get_gen_mask
import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from torch.nn.functional import scaled_dot_product_attention
from utils import assert_close


def get_qkv(B=4, H=4, S=32, D=16):
    torch.random.manual_seed(42)
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H // 2, S, D)
    v = torch.randn(B, H // 2, S, D)
    return q, k, v


def test_prefill_mask():
    q, k, v = get_qkv()
    B, H, S, D = q.shape
    ref_output = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    seqlens = torch.randint(S // 2, S, (B,))

    mask = make_prefill_mask(seqlens, S, S, compile=False)
    output = flex_attention(q, k, v, block_mask=mask, enable_gqa=True)
    for i in range(B):
        ilen = seqlens[i]
        assert_close(output[i, :, :ilen], ref_output[i, :, :ilen])


def test_gen_mask():
    q, k, v = get_qkv()
    device = q.device
    B, H, S, D = q.shape
    ref_output = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    offsets = torch.randint(1, S - 1, (B,), device=device, dtype=torch.int)
    mask = get_gen_mask(offsets, S, BLOCK_SIZE=S // 8)
    reference_mask = create_block_mask(
        offset_mask_mod(offsets), B, 1, 1, S, BLOCK_SIZE=S // 4, device=device
    )

    k_mask, v_mask = k.clone(), v.clone()
    q_mask = torch.zeros(B, H, 1, D, device=q.device)
    for b in range(B):
        j = offsets[b]
        k_mask[b, :, j + 1 :] = 1e-9
        v_mask[b, :, j + 1 :] = 1e-9
        q_mask[b, :, 0] = q[b, :, j]

    output1 = flex_attention(
        q_mask, k_mask, v_mask, block_mask=reference_mask, enable_gqa=True
    )
    output2 = flex_attention(q_mask, k_mask, v_mask, block_mask=mask, enable_gqa=True)

    for i in range(B):
        j = offsets[i]
        assert_close(output1[i], ref_output[i, :, j : j + 1])
        assert_close(output2[i], ref_output[i, :, j : j + 1])
