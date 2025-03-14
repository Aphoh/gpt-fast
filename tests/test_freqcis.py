import torch

from gpt_fast.model import apply_rotary_emb, precompute_freqs_cis
from gpt_fast.util import input_pos_from_start_inds


def test_apply_rotary_emb():
    seq_len = 128
    head_dim = 4
    num_heads = 2
    dtype = torch.bfloat16
    B = 4

    start_inds = torch.arange(B)
    input_pos = input_pos_from_start_inds(start_inds, seq_len)

    freqs_cis = precompute_freqs_cis(seq_len=seq_len, n_elem=head_dim, dtype=dtype)
    x = torch.ones(B, seq_len, num_heads, head_dim)

    freqs_cis_unbatched = freqs_cis[torch.arange(seq_len)]
    x_unbatched = apply_rotary_emb(x, freqs_cis_unbatched)
    for i in range(1, B):
        assert torch.all(x_unbatched[0] == x_unbatched[i])

    freqs_cis_to_apply = freqs_cis[input_pos]
    res = apply_rotary_emb(x, freqs_cis_to_apply)
    for i in range(1, B):
        assert torch.all(res[i, i:] == res[0, :-i])
