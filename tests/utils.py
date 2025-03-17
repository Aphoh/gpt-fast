import torch


def assert_close(a, b, atol=1e-5, rtol=1e-5):
    abs_diff = torch.abs(a - b).max()
    assert torch.allclose(a, b, atol=atol, rtol=rtol), f"abs_diff={abs_diff}"
