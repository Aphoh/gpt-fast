import pytest
import torch
from gpt_fast.generate import SamplingConfig, generate
from gpt_fast.model import Transformer, ModelArgs
from gpt_fast.ckpt_utils import convert_state_dict
from transformers import LlamaForCausalLM, LlamaConfig, GenerationConfig
from torch.testing import assert_close


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


def small_configs():
    config = LlamaConfig(
        vocab_size=1000,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        initializer_range=0.1,
    )

    model_args = ModelArgs(
        vocab_size=1000,
        dim=16,
        intermediate_size=32,
        n_head=4,
        n_local_heads=2,
        n_layer=6,
        block_size=1024,
    )
    return config, model_args


@pytest.fixture
def reference_model():
    torch.random.manual_seed(42)
    return LlamaForCausalLM(small_configs()[0])


@pytest.fixture
def small_model(reference_model):
    ref_state_dict = reference_model.state_dict()
    _, config = small_configs()
    state_dict = convert_state_dict(config, ref_state_dict)
    tformer = Transformer(config)
    tformer.load_state_dict(state_dict)
    return tformer


@skip_if_no_cuda
def test_llama_decode():
    with torch.device("cuda"):
        name = "unsloth/Llama-3.2-1B-Instruct"
        reference_model = LlamaForCausalLM.from_pretrained(name)
        ref_state_dict = reference_model.state_dict()
        config = ModelArgs.from_name(name)
        state_dict = convert_state_dict(config, ref_state_dict)
        our_model = Transformer(config)
        our_model.load_state_dict(state_dict)
        test_decode_consistency(our_model, reference_model)


@torch.inference_mode()
def test_generate_long_seqlen(small_model: Transformer):
    torch.random.manual_seed(42)
    B = 2
    S = 32
    max_new_tokens = 16
    seqlens = torch.tensor([16, 31])
    input_ids = torch.randint(0, small_model.config.vocab_size, (B, S))
    input_ids[0, 16:] = -1
    # one very long sequence near the max length
    # one shorter sequence
    small_model.setup_caches(B, S)
    output = generate(
        small_model,
        input_ids=input_ids,
        seqlens=seqlens,
        max_seqlen=S,
        max_new_tokens=max_new_tokens,
        device=input_ids.device,
        compile=False,
        sampling=SamplingConfig(top_k=None, temperature=0.0),
    )
    assert output.shape == (B, S)
    # the first sequence should have new tokens
    assert (output[0, 16:] != -1).all()
    # the first sequence should be unchanged other than the later tokens
    assert (output[0, :16] == input_ids[0, :16]).all()
    # the second sequence should be unchanged other than the last token
    assert (output[1, :-1] == input_ids[1, :-1]).all()


@torch.inference_mode()
def test_decode_consistency(
    small_model: Transformer,
    reference_model: LlamaForCausalLM,
):
    torch.random.manual_seed(42)
    # Input setup
    B = 4
    PROMPT_S = 5
    PREFILL_S = 16
    GEN_S = 32
    ref_input_ids = torch.randint(0, small_model.config.vocab_size, (1, PROMPT_S))
    generation_config = GenerationConfig(
        num_beams=1,
        do_sample=False,
        max_new_tokens=GEN_S - ref_input_ids.shape[1],
    )
    attention_mask = torch.ones_like(ref_input_ids, dtype=torch.int)
    ref_output = reference_model.generate(
        ref_input_ids,
        generation_config,
        attention_mask=attention_mask,
    ).to(torch.int)

    input_ids = torch.zeros((B, PREFILL_S), dtype=torch.int)
    seqlens = torch.randint(PROMPT_S + 1, PREFILL_S, (B,))
    for b in range(B):
        input_ids[b, : seqlens[b]] = ref_output[0, : seqlens[b]]

    small_model.setup_caches(B, GEN_S)

    max_new_tokens = GEN_S - PREFILL_S
    output = generate(
        small_model,
        input_ids=input_ids,
        seqlens=seqlens,
        max_seqlen=GEN_S,
        max_new_tokens=max_new_tokens,
        device=input_ids.device,
        compile=False,
        sampling=SamplingConfig(top_k=None, temperature=0.0),
    )

    for b in range(0, B):
        gen_seqlen = seqlens[b] + max_new_tokens
        assert_close(
            output[b, :gen_seqlen],
            ref_output[0, :gen_seqlen],
        )


@torch.inference_mode()
def test_stopping_condition(small_model: Transformer):
    torch.random.manual_seed(42)
    # Input setup - similar to test_decode_consistency
    B = 4
    PREFILL_S = 16
    GEN_S = 32

    input_ids = torch.zeros((B, PREFILL_S), dtype=torch.int)
    seqlens = torch.randint(2, PREFILL_S, (B,))

    small_model.setup_caches(B, GEN_S)

    max_new_tokens = GEN_S - PREFILL_S

    # First generate without stopping condition
    output_no_stop = generate(
        small_model,
        input_ids=input_ids,
        seqlens=seqlens,
        max_seqlen=GEN_S,
        max_new_tokens=max_new_tokens,
        device=input_ids.device,
        compile=False,
        sampling=SamplingConfig(top_k=None, temperature=0.0),
    )

    stop_inds = GEN_S - torch.arange(B) - 1
    stop_tokens = torch.zeros(B, dtype=torch.int, device=input_ids.device)
    for b in range(B):
        stop_tokens[b] = output_no_stop[b, stop_inds[b]]

    # Create a stopping condition that stops when it sees the specific token
    def custom_stopping_condition(tokens):
        # tokens shape is [B, tokens_generated_so_far]
        # Return a boolean tensor of shape [B]
        # Check if the token at stop_gen_pos position matches our stop token
        return tokens[:, -1] == stop_tokens

    # Generate again with the stopping condition
    small_model.setup_caches(B, GEN_S)  # Reset caches
    output_with_stop = generate(
        small_model,
        input_ids=input_ids,
        seqlens=seqlens,
        max_seqlen=GEN_S,
        max_new_tokens=max_new_tokens,
        device=input_ids.device,
        compile=False,
        sampling=SamplingConfig(top_k=None, temperature=0.0),
        stopping_condition=custom_stopping_condition,
    )

    # Check the results
    for b in range(B):
        end_idx = stop_inds[b] + 1
        # Check that the outputs match up to the stopping point
        assert torch.all(output_with_stop[b, :end_idx] == output_no_stop[b, :end_idx])

        # Check that all tokens after the stopping point are -1
        assert torch.all(output_with_stop[b, end_idx:] == -1)
