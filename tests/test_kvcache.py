import pytest
import torch
from gpt_fast.model import KVCache


# Fixture to set up the KVCache for each test
@pytest.fixture
def kv_cache():
    max_batch_size = 4
    max_seq_length = 16
    n_heads = 2
    head_dim = 8
    dtype = torch.float32  # Using float32 for testing precision

    # Create a KVCache instance for testing
    cache = KVCache(
        max_batch_size=max_batch_size,
        max_seq_length=max_seq_length,
        n_heads=n_heads,
        head_dim=head_dim,
        dtype=dtype,
    )

    # Initialize cache with zeros
    cache.k_cache.fill_(0.0)
    cache.v_cache.fill_(0.0)

    return cache


def test_basic_update(kv_cache):
    """Test basic update functionality with valid indices."""
    batch_size = kv_cache.k_cache.shape[0]
    seq_length = 4
    n_heads = kv_cache.k_cache.shape[1]
    head_dim = kv_cache.k_cache.shape[3]

    # Create recognizable values for testing
    k_val = torch.arange(
        1,
        batch_size * n_heads * seq_length * head_dim + 1,
        dtype=kv_cache.k_cache.dtype,
    ).reshape(batch_size, n_heads, seq_length, head_dim)
    v_val = k_val + 100  # Different values for v_cache

    # Define positions to update
    offset = 2

    # Update cache
    k_out, v_out = kv_cache.update(torch.tensor([offset]), k_val, v_val)

    # Verify results
    for b in range(batch_size):
        for pos_idx in range(offset, offset + seq_length):
            # Check that values were properly placed in cache
            assert torch.allclose(k_out[b, :, pos_idx], k_val[b, :, pos_idx - offset])
            assert torch.allclose(v_out[b, :, pos_idx], v_val[b, :, pos_idx - offset])
