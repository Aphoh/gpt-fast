import pytest
import torch
from gpt_fast.generate import generate
from gpt_fast.model import Transformer, ModelArgs


@pytest.fixture
def test_model():
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


def test_generate(test_model: Transformer):
    # Input setup
    batch_size = 4
    prompt_seq_length = 12
    input_ids = torch.ones((batch_size, prompt_seq_length), dtype=torch.int)
    start_inds = torch.arange(batch_size, dtype=torch.int)
    max_seq_length = 128
    max_new_tokens = 16
    device = torch.device("cpu")

    test_model.setup_caches(batch_size, max_seq_length)

    # Call generate function
    with torch.no_grad():
        output_ids = generate(
            model=test_model,
            input_ids=input_ids,
            start_inds=start_inds,
            max_seq_length=max_seq_length,
            max_new_tokens=max_new_tokens,
            device=device,
            compile=False,
            temperature=1.0,
            top_k=None,
        )

    # Verify results
    assert output_ids.shape == (batch_size, max_new_tokens)
    # With our test model, each token should follow a pattern
    # The exact values will depend on our mock model implementation
    assert not torch.all(output_ids == -1), "Output should not be all padding"
