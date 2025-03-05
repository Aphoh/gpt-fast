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
    batch_size = 2
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
    input_pos = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.long)

    # Update cache
    k_out, v_out = kv_cache.update(input_pos, k_val, v_val)

    # Verify results
    for b in range(batch_size):
        for pos_idx in range(seq_length):
            pos = input_pos[b, pos_idx].item()
            # Check that values were properly placed in cache
            assert torch.allclose(k_out[b, :, pos], k_val[b, :, pos_idx])
            assert torch.allclose(v_out[b, :, pos], v_val[b, :, pos_idx])


def test_update_with_negative_indices(kv_cache):
    """Test that positions with -1 are properly ignored."""
    batch_size = 2
    seq_length = 4
    n_heads = kv_cache.k_cache.shape[1]
    head_dim = kv_cache.k_cache.shape[3]

    # Create test values
    k_val = (
        torch.ones(
            (batch_size, n_heads, seq_length, head_dim), dtype=kv_cache.k_cache.dtype
        )
        * 5.0
    )
    v_val = (
        torch.ones(
            (batch_size, n_heads, seq_length, head_dim), dtype=kv_cache.v_cache.dtype
        )
        * 10.0
    )

    # Define positions with some -1 values
    input_pos = torch.tensor(
        [
            [0, 1, -1, 3],  # Batch 0: Skip position 2
            [-1, 5, 6, -1],  # Batch 1: Skip positions 0 and 3
        ],
        dtype=torch.long,
    )

    # Update cache
    k_out, v_out = kv_cache.update(input_pos, k_val, v_val)

    # Check batch 0
    assert torch.allclose(k_out[0, :, 0], k_val[0, :, 0])  # Updated
    assert torch.allclose(k_out[0, :, 1], k_val[0, :, 1])  # Updated
    assert torch.allclose(
        k_out[0, :, 2], torch.zeros_like(k_out[0, :, 2])
    )  # Not updated (-1)
    assert torch.allclose(k_out[0, :, 3], k_val[0, :, 3])  # Updated

    # Check batch 1
    assert torch.allclose(
        k_out[1, :, 0], torch.zeros_like(k_out[1, :, 0])
    )  # Not updated (-1)
    assert torch.allclose(k_out[1, :, 5], k_val[1, :, 1])  # Updated
    assert torch.allclose(k_out[1, :, 6], k_val[1, :, 2])  # Updated

    # Similar checks for v_cache
    assert torch.allclose(v_out[0, :, 0], v_val[0, :, 0])
    assert torch.allclose(v_out[0, :, 2], torch.zeros_like(v_out[0, :, 2]))
    assert torch.allclose(v_out[1, :, 5], v_val[1, :, 1])


def test_different_indices_per_batch(kv_cache):
    """Test that each batch can have completely different indices."""
    batch_size = 3
    seq_length = 2
    n_heads = kv_cache.k_cache.shape[1]
    head_dim = kv_cache.k_cache.shape[3]

    # Create test values with recognizable pattern
    k_val = torch.tensor(
        range(1, batch_size * n_heads * seq_length * head_dim + 1),
        dtype=kv_cache.k_cache.dtype,
    ).reshape(batch_size, n_heads, seq_length, head_dim)
    v_val = k_val * 2

    # Define completely different positions for each batch
    input_pos = torch.tensor(
        [
            [0, 1],  # Batch 0: positions 0, 1
            [10, 11],  # Batch 1: positions 10, 11
            [15, 13],  # Batch 2: positions 15, 13 (out of order)
        ],
        dtype=torch.long,
    )

    # Update cache
    k_out, v_out = kv_cache.update(input_pos, k_val, v_val)

    # Verify results
    assert torch.allclose(k_out[0, :, 0], k_val[0, :, 0])
    assert torch.allclose(k_out[1, :, 10], k_val[1, :, 0])
    assert torch.allclose(k_out[2, :, 15], k_val[2, :, 0])
    assert torch.allclose(k_out[2, :, 13], k_val[2, :, 1])

    assert torch.allclose(v_out[0, :, 1], v_val[0, :, 1])
    assert torch.allclose(v_out[1, :, 11], v_val[1, :, 1])


def test_all_negative_indices(kv_cache):
    """Test the case where all indices are -1 (no updates)."""
    batch_size = 2
    seq_length = 3
    n_heads = kv_cache.k_cache.shape[1]
    head_dim = kv_cache.k_cache.shape[3]

    # Create test values
    k_val = (
        torch.ones(
            (batch_size, n_heads, seq_length, head_dim), dtype=kv_cache.k_cache.dtype
        )
        * 5.0
    )
    v_val = (
        torch.ones(
            (batch_size, n_heads, seq_length, head_dim), dtype=kv_cache.v_cache.dtype
        )
        * 10.0
    )

    # All positions are -1
    input_pos = torch.full((batch_size, seq_length), -1, dtype=torch.long)

    # Set initial cache values for verification
    kv_cache.k_cache.fill_(1.0)
    kv_cache.v_cache.fill_(2.0)

    # Update cache
    k_out, v_out = kv_cache.update(input_pos, k_val, v_val)

    # Verify cache is unchanged
    assert torch.allclose(k_out, kv_cache.k_cache)
    assert torch.allclose(v_out, kv_cache.v_cache)


def test_update_multiple_times(kv_cache):
    """Test updating the cache multiple times, ensuring updates are cumulative."""
    batch_size = 2
    seq_length = 2
    n_heads = kv_cache.k_cache.shape[1]
    head_dim = kv_cache.k_cache.shape[3]

    # First update data
    k_val1 = torch.full(
        (batch_size, n_heads, seq_length, head_dim), 3.0, dtype=kv_cache.k_cache.dtype
    )
    v_val1 = torch.full(
        (batch_size, n_heads, seq_length, head_dim), 4.0, dtype=kv_cache.v_cache.dtype
    )
    input_pos1 = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)

    # Second update data (different positions)
    k_val2 = torch.full(
        (batch_size, n_heads, seq_length, head_dim), 5.0, dtype=kv_cache.k_cache.dtype
    )
    v_val2 = torch.full(
        (batch_size, n_heads, seq_length, head_dim), 6.0, dtype=kv_cache.v_cache.dtype
    )
    input_pos2 = torch.tensor([[2, 3], [0, 1]], dtype=torch.long)

    # First update
    k_out1, v_out1 = kv_cache.update(input_pos1, k_val1, v_val1)

    # Use first update results as new cache
    kv_cache.k_cache.copy_(k_out1)
    kv_cache.v_cache.copy_(v_out1)

    # Second update
    k_out2, v_out2 = kv_cache.update(input_pos2, k_val2, v_val2)

    # Verify final state contains both updates
    assert torch.allclose(k_out2[0, :, 0], k_val1[0, :, 0])  # From first update
    assert torch.allclose(k_out2[0, :, 2], k_val2[0, :, 0])  # From second update
    assert torch.allclose(k_out2[1, :, 0], k_val2[1, :, 0])  # From second update
    assert torch.allclose(k_out2[1, :, 2], k_val1[1, :, 0])  # From first update
