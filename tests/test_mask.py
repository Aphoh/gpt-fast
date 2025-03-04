from gpt_fast.mask_utils import (
    make_base_gen_mask,
    make_prefill_mask,
)
import torch


def test_left_pad_mask():
    device = torch.device("cpu")
    B = 4
    S = 128
    block_size = 4
    start_inds = torch.arange(B) * block_size

    # base gen mask
    mask = make_base_gen_mask(
        start_inds,
        max_seq_length=S,
        device=device,
        compile=False,
        BLOCK_SIZE=(block_size, block_size),
    )

    def check_mask(mask):
        print(mask.to_string())
        for i in range(B):
            assert (
                mask.full_kv_num_blocks[i, 0, : i + 1] == 0
            ).all()  # the +1 is just including the 0 index
            assert (mask.full_q_num_blocks[i, 0, :i] == 0).all()

    check_mask(mask)

    input_pos = torch.arange(S // 2)
    mask = make_prefill_mask(
        start_inds,
        input_pos,
        max_seq_length=S,
        device=device,
        compile=False,
        BLOCK_SIZE=(block_size, block_size),
    )
    check_mask(mask)
