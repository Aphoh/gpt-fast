import pytest
from gpt_fast.mask_utils import left_pad_mask_mod, make_base_mask, get_gen_submask
import torch
from torch.nn.attention.flex_attention import flex_attention
from torch.nn.functional import scaled_dot_product_attention

from gpt_fast.util import flex_attention_maybe_pad
from tests.util import skip_if_no_cuda


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


def test_basic_reference():
    device = torch.device("cuda")
    B = 8
    S = 16
    emb_dim = 16
    full_mask = torch.tril(torch.ones(S, S, dtype=torch.bool, device=device))
    ref_mask = full_mask[None, None, ...].expand(B, 1, -1, -1).clone()
    qkv = torch.randn(B, 1, S, emb_dim, device=device)
    ref_out = scaled_dot_product_attention(qkv, qkv, qkv, attn_mask=ref_mask, is_causal=True)
    mask = make_base_mask(B, S, S, device=device, compile=False)
    flex_out = flex_attention(qkv, qkv, qkv, block_mask=mask)
    assert torch.allclose(ref_out, flex_out, atol=1e-2, rtol=1e-2)


def ref_attn_with_start_inds(q, k, v, start_inds):
    B, num_heads, seq_len, dim = q.shape
    output = torch.zeros_like(q)
    for b in range(B):
        si = start_inds[b]
        q_b, k_b, v_b = q[b, :, si:], k[b, :, si:], v[b, :, si:]
        attn_out = scaled_dot_product_attention(q_b, k_b, v_b, is_causal=True)
        output[b, :, start_inds[b]:] = attn_out[0]
    return output



@pytest.mark.parametrize("compile", [True, False])
@skip_if_no_cuda
def test_left_pad_mask_big_padding(compile):
    with torch.device("cuda"):
        device = torch.device("cuda")
        B = 8
        BASE_S = 64
        pad_left = 16 
        FULL_S = pad_left + BASE_S
        emb_dim = 16
        start_inds = pad_left + torch.arange(B)

        all_base_mask = make_base_mask(B, FULL_S, FULL_S, device=device, compile=compile)
        all_base_mask.mask_mod = left_pad_mask_mod(start_inds, torch.tensor([0]))

        gen_base = make_base_mask(B, FULL_S, FULL_S, device=device, compile=compile)
        fx = torch.compile(flex_attention) if compile else flex_attention

        with torch.no_grad():
            qkv = torch.randn(B, 1, FULL_S, emb_dim, device=device)
            ref_out = ref_attn_with_start_inds(qkv, qkv, qkv, start_inds)
            all_out = fx(qkv, qkv, qkv, block_mask=all_base_mask)
            assert torch.allclose(all_out[0, :, pad_left:], ref_out[0, :, pad_left:], atol=1e-5, rtol=1e-5)

            for query_idx in range(pad_left, FULL_S):
                gen_submask = get_gen_submask(gen_base, query_idx)
                gen_submask.mask_mod = left_pad_mask_mod(
                    start_inds, torch.tensor([query_idx])
                )
                gen_out = fx(
                    qkv[:, :, query_idx : query_idx + 1], qkv, qkv, block_mask=gen_submask
                )
                assert torch.allclose(all_out[:, :, query_idx], gen_out[:, :, 0], atol=1e-2, rtol=1e-2)


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

    output = flex_attention_maybe_pad(
        qkv.clone(), qkv.clone(), qkv.clone(), block_mask=mask
    )

    assert torch.allclose(output[0, :, :-1], output[1, :, 1:])
