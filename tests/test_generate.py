import pytest
import torch
from gpt_fast.generate import SamplingConfig, generate
from gpt_fast.model import Transformer, ModelArgs
from gpt_fast.ckpt_utils import convert_state_dict
from transformers import LlamaForCausalLM, LlamaConfig, GenerationConfig

from util import skip_if_no_cuda


def small_configs():
    config = LlamaConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=32,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        initializer_range=0.1,
        tie_word_embeddings=True,
    )

    model_args = ModelArgs(
        vocab_size=1000,
        dim=64,
        intermediate_size=32,
        n_head=4,
        n_local_heads=2,
        n_layer=6,
        block_size=1024,
        tie_embedding_weights=True,
    )
    return config, model_args


def get_reference_model():
    torch.random.manual_seed(42)
    return LlamaForCausalLM(small_configs()[0])


def get_small_model(reference_model):
    ref_state_dict = reference_model.state_dict()
    _, config = small_configs()
    state_dict = convert_state_dict(config, ref_state_dict)
    if config.tie_embedding_weights:
        state_dict["output.weight"] = state_dict["tok_embeddings.weight"]
    tformer = Transformer(config)
    tformer.load_state_dict(state_dict)
    return tformer

@pytest.mark.parametrize("compile", [True, False])
@pytest.mark.parametrize("left_pad", [0, 256])
@skip_if_no_cuda
def test_small_model_decode(compile, left_pad):
    with torch.device("cuda"):
        reference_model = get_reference_model().to(device="cuda", dtype=torch.float32)
        small_model = get_small_model(reference_model).to(device="cuda", dtype=torch.float32)
        do_test_decode_consistency(small_model, reference_model, left_pad=left_pad, compile=compile)



@skip_if_no_cuda
def test_llama_decode():
    with torch.device("cuda"):
        name = "unsloth/Llama-3.2-1B-Instruct"
        reference_model = LlamaForCausalLM.from_pretrained(
            name, torch_dtype=torch.bfloat16
        )
        ref_state_dict = reference_model.state_dict()
        config = ModelArgs.from_name(name)
        state_dict = convert_state_dict(config, ref_state_dict)
        our_model = Transformer(config).to(torch.bfloat16)
        if our_model.config.tie_embedding_weights:
            state_dict["output.weight"] = state_dict["tok_embeddings.weight"]
        our_model.load_state_dict(state_dict)
        do_test_decode_consistency(our_model, reference_model, compile=True)

@torch.inference_mode()
def do_test_decode_consistency(
    model: Transformer, reference_model: LlamaForCausalLM, compile=False, left_pad=0
):
    # Input setup
    batch_size = 4
    prompt_seq_length = 16
    max_seq_length = 24
    ref_input_ids = torch.randint(0, model.config.vocab_size, (1, 5))
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

    prompt_len_with_pad = prompt_seq_length + left_pad
    max_seq_len_with_pad = max_seq_length + left_pad
    input_ids = torch.zeros((batch_size, prompt_len_with_pad), dtype=torch.int)
    input_ids[0, left_pad:] = ref_output[0, :prompt_seq_length]
    for i in range(1, batch_size):
        input_ids[i, left_pad + i:] = input_ids[0, left_pad:-i]
    start_inds = left_pad + torch.arange(batch_size, dtype=torch.int)

    model.setup_caches(batch_size, max_seq_len_with_pad)

    output = generate(
        model,
        input_ids=input_ids,
        start_inds=start_inds,
        max_seq_length=max_seq_len_with_pad,
        max_new_tokens=max_seq_len_with_pad - prompt_len_with_pad,
        device=input_ids.device,
        compile=compile,
        sampling=SamplingConfig(top_k=None, temperature=0.0),
    )

    print("REF OUT  ", ref_output[0].cpu())
    for b in range(batch_size):
        print(f"OUR OUT {b}", output[b, left_pad + b:].cpu())
    assert (output[0, left_pad:].cpu() == ref_output[0].cpu()).all()
    for b in range(1, batch_size):
        assert (output[b, left_pad + b:].cpu() == ref_output[0, :-b].cpu()).all(), (
            f"Output mismatch, at batch {b}"
        )
