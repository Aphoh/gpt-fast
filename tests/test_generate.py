import pytest
import torch
from gpt_fast.generate import decode_n_tokens, generate, prefill
from gpt_fast.mask_utils import make_base_mask
from gpt_fast.model import Transformer, ModelArgs
from gpt_fast.util import input_pos_from_start_inds


@pytest.fixture
def small_model():
    return Transformer(
        ModelArgs(
            block_size=256,
            vocab_size=1000,
            n_layer=2,
            n_head=2,
            dim=64,
            intermediate_size=256,
            glu=True,
            tie_embedding_weights=True,
        )
    )


def test_prefill(small_model: Transformer):
    # Input setup
    batch_size = 4
    prompt_seq_length = 12
    input_ids = torch.ones((batch_size, prompt_seq_length), dtype=torch.int)
    start_inds = torch.arange(batch_size, dtype=torch.int)
    input_pos = input_pos_from_start_inds(start_inds, prompt_seq_length)
    max_seq_length = 128

    small_model.setup_caches(batch_size, max_seq_length)

    # Call generate function
    with torch.no_grad():
        _ = prefill(
            model=small_model,
            x=input_ids,
            input_pos=input_pos,
            start_inds=start_inds,
            max_seq_length=max_seq_length,
            compile=False,
        )


def test_decode_n(small_model: Transformer):
    # Input setup
    batch_size = 4
    prompt_seq_length = 12
    max_seq_length = 128
    input_ids = torch.ones(batch_size, dtype=torch.int)
    start_inds = torch.arange(batch_size, dtype=torch.int)
    input_pos = input_pos_from_start_inds(start_inds, prompt_seq_length)[:, -1]

    small_model.setup_caches(batch_size, max_seq_length)

    base_mask = make_base_mask(
        batch_size, max_seq_length, max_seq_length, device=start_inds.device
    )

    # Call generate function
    with torch.no_grad():
        decode_n_tokens(
            base_mask=base_mask,
            query_pos=prompt_seq_length,
            model=small_model,
            start_inds=start_inds,
            cur_token=input_ids,
            input_pos=input_pos,
            compile=False,
            max_new_tokens=4,
        )


def test_generate(small_model: Transformer):
    # Input setup
    device = torch.device("cpu")
    batch_size = 4
    prompt_seq_length = 12
    input_ids = torch.ones((batch_size, prompt_seq_length), dtype=torch.int)
    start_inds = torch.arange(batch_size, dtype=torch.int)
    max_seq_length = 128
    max_new_tokens = 16

    small_model.setup_caches(batch_size, max_seq_length)

    # Call generate function
    with torch.no_grad():
        output_ids = generate(
            model=small_model,
            input_ids=input_ids,
            start_inds=start_inds,
            max_seq_length=max_seq_length,
            max_new_tokens=max_new_tokens,
            device=device,
            compile=False,
        )

    # Verify results
    assert output_ids.shape == (batch_size, prompt_seq_length + max_new_tokens)
