import torch
from gpt_fast.util import expand_router_probs, keep_topk
from torch.testing import assert_close


def test_keep_topk():
    inp = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.4, 0.3, 0.2, 0.1],
        ]
    )
    assert_close(
        keep_topk(inp, 2, dim=-1),
        torch.tensor(
            [
                [0.0, 0.0, 0.3, 0.4],
                [0.4, 0.3, 0.0, 0.0],
            ]
        ),
    )


# fmt: off
def test_expand_router_probs():
    inp = torch.tensor([
        [0.1, 0.2, 0.3, 0.4],
        [0.4, 0.3, 0.2, 0.1],
    ])
    inp = keep_topk(inp, 2, dim=-1)
    expanded = expand_router_probs(inp, 4)
    assert expanded.shape == (2, 16)
    assert_close(
        expanded,
        torch.tensor([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4, 0.4], 
            [0.4, 0.4, 0.4, 0.4, 0.3, 0.3, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ])
    )
# fmt: on
