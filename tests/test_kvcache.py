import pytest
import torch
from gpt_fast.model import KVCache


# Fixture to set up the KVCache for each test
@pytest.fixture
def kv_cache() -> KVCache:
    max_batch_size = 4
    max_seqlen = 16
    n_heads = 2
    head_dim = 8
    dtype = torch.float32  # Using float32 for testing precision

    # Create a KVCache instance for testing
    cache = KVCache(
        max_batch_size=max_batch_size,
        max_seqlen=max_seqlen,
        n_heads=n_heads,
        head_dim=head_dim,
        dtype=dtype,
    )

    # Initialize cache with zeros
    cache.k_cache.fill_(0.0)
    cache.v_cache.fill_(0.0)

    return cache


@pytest.fixture
def k_vals(kv_cache):
    B, S, H, D = kv_cache.k_cache.shape

    # Create recognizable values for testing
    return torch.arange(
        1,
        B * S * H * D + 1,
        dtype=kv_cache.k_cache.dtype,
    ).reshape(B, S, H, D)


@pytest.fixture
def v_vals(k_vals):
    return k_vals + 1000


def test_basic_update(kv_cache: KVCache, k_vals, v_vals):
    """Test basic update functionality with valid indices."""

    B, H, S, D = k_vals.shape

    input_pos = torch.arange(S)[None, :].expand(B, -1)
    kv_cache.update(input_pos, k_vals, v_vals)

    assert torch.allclose(kv_cache.k_cache, k_vals)
    assert torch.allclose(kv_cache.v_cache, v_vals)


def test_update_with_narrow_inds(kv_cache: KVCache, k_vals, v_vals):
    """Test basic update functionality with valid indices."""

    B, H, S, D = k_vals.shape

    input_pos = torch.randint(0, S, (B, 1))
    k_vals_i = torch.zeros(B, H, 1, D)
    v_vals_i = torch.zeros(B, H, 1, D)
    for b in range(B):
        k_vals_i[b] = k_vals[b, :, input_pos[b]]
        v_vals_i[b] = v_vals[b, :, input_pos[b]]
    kv_cache.update(input_pos, k_vals_i, v_vals_i)

    for b in range(B):
        assert torch.allclose(kv_cache.k_cache[b, :, input_pos[b]], k_vals_i[b])
        assert torch.allclose(kv_cache.v_cache[b, :, input_pos[b]], v_vals_i[b])
