import pytest
import torch
from gpt_fast.generate import SamplingConfig, generate
from gpt_fast.model import Transformer, ModelArgs
from gpt_fast.ckpt_utils import convert_state_dict
from transformers import LlamaForCausalLM, LlamaConfig, GenerationConfig


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


@pytest.mark.slow
def test_llama_decode():
    with torch.device("cuda"):
        name = "unsloth/Llama-3.2-1B-Instruct"
        reference_model = LlamaForCausalLM.from_pretrained(name)
        ref_state_dict = reference_model.state_dict()
        config = ModelArgs.from_name(name)
        state_dict = convert_state_dict(config, ref_state_dict)
        our_model = Transformer(config)
        our_model.load_state_dict(state_dict)
        test_decode_consistency(our_model, reference_model, compile=True)


@torch.inference_mode()
def test_decode_consistency(
    small_model: Transformer, reference_model: LlamaForCausalLM, compile=False
):
    # Input setup
    batch_size = 4
    prompt_seq_length = 16
    max_seq_length = 32
    ref_input_ids = torch.randint(0, small_model.config.vocab_size, (1, 5))
    generation_config = GenerationConfig(
        num_beams=1,
        do_sample=False,
        max_new_tokens=max_seq_length - ref_input_ids.shape[1],
    )
    attention_mask = torch.ones_like(ref_input_ids, dtype=torch.int)
    ref_output = reference_model.generate(
        ref_input_ids,
        generation_config,
        attention_mask=attention_mask,
    )

    input_ids = torch.zeros((batch_size, prompt_seq_length), dtype=torch.int)
    input_ids[0] = ref_output[0, :prompt_seq_length]
    for i in range(1, batch_size):
        input_ids[i, i:] = input_ids[0, :-i]
    start_inds = torch.arange(batch_size, dtype=torch.int)

    small_model.setup_caches(batch_size, max_seq_length)

    output = generate(
        small_model,
        input_ids=input_ids,
        start_inds=start_inds,
        max_seq_length=max_seq_length,
        max_new_tokens=max_seq_length - prompt_seq_length,
        device=input_ids.device,
        compile=compile,
        sampling=SamplingConfig(top_k=None, temperature=0.0),
    )

    assert (output[0].cpu() == ref_output[0].cpu()).all()
    for b in range(1, batch_size):
        assert (output[b, b:].cpu() == ref_output[0, :-b].cpu()).all(), (
            f"Output mismatch, at batch {b}"
        )
