import pytest
import torch


def skip_if_no_cuda(func):
    """Decorator to skip tests when CUDA is not available.

    Example:
        @skip_if_no_cuda
        def test_gpu_function():
            # This test will be skipped if no GPU is available
            assert torch.cuda.is_available()
    """
    return pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available, skipping GPU test"
    )(func)
