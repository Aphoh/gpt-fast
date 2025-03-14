import tempfile
import pytest
import torch
from gpt_fast.generate import SamplingConfig, generate
from gpt_fast.model import Transformer, ModelArgs
from gpt_fast.ckpt_utils import convert_state_dict
from transformers import LlamaForCausalLM, LlamaConfig, GenerationConfig

from gpt_fast.util import load_model
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


def get_small_model_meta_load(reference_model):
    _, config = small_configs()
    state_dict = reference_model.state_dict()
    converted = convert_state_dict(config, state_dict)
    if config.tie_embedding_weights:
        converted["output.weight"] = converted["tok_embeddings.weight"]
    with torch.device("meta"):
        model = Transformer(config)
    model.load_state_dict(converted, assign=True)
    return model


def get_small_model_disk_load(reference_model, mmap=False):
    with tempfile.TemporaryDirectory() as tmpdir:
        _, config = small_configs()
        state_dict = reference_model.state_dict()
        converted = convert_state_dict(config, state_dict)
        out_file = tmpdir + "/model.pth"
        torch.save(converted, out_file)
        loaded_state_dict = torch.load(out_file, weights_only=True, mmap=mmap)
        if config.tie_embedding_weights:
            loaded_state_dict["output.weight"] = loaded_state_dict[
                "tok_embeddings.weight"
            ]
        with torch.device("meta"):
            model = Transformer(config)
        model.load_state_dict(loaded_state_dict, assign=True)
    return model


def get_model_load_model_method(reference_model, config=None):
    with tempfile.TemporaryDirectory() as tmpdir:
        if config is None:
            _, config = small_configs()
        state_dict = reference_model.state_dict()
        converted = convert_state_dict(config, state_dict)
        out_file = tmpdir + "/model.pth"
        torch.save(converted, out_file)
        with torch.device("meta"):
            model = Transformer(config)
        model = load_model(model, out_file, precision=torch.bfloat16, device="cuda")
    return model


def test_small_model_decode():
    reference_model = get_reference_model().cpu()
    small_model = get_small_model(reference_model).cpu()
    do_test_decode_consistency(small_model, reference_model)


@skip_if_no_cuda
def test_small_model_decode_compile_fp32():
    reference_model = get_reference_model()
    small_model = get_small_model(reference_model)
    with torch.device("cuda"):
        small_model = small_model.cuda()
        reference_model = reference_model.cuda()
        for w in small_model.parameters():
            assert w.dtype == torch.float32
        for w in reference_model.parameters():
            assert w.dtype == torch.float32
        do_test_decode_consistency(small_model, reference_model, compile=True)


@skip_if_no_cuda
def test_small_model_decode_compile_bf16():
    reference_model = get_reference_model()
    small_model = get_small_model(reference_model)
    with torch.device("cuda"):
        small_model = small_model.to(torch.bfloat16).cuda()
        reference_model = reference_model.to(torch.bfloat16).cuda()
        for w in small_model.parameters():
            assert w.dtype == torch.bfloat16
        for w in reference_model.parameters():
            assert w.dtype == torch.bfloat16
        do_test_decode_consistency(small_model, reference_model, compile=True)


@skip_if_no_cuda
def test_small_model_decode_compile_meta_load():
    reference_model = get_reference_model().to(device="cuda", dtype=torch.bfloat16)
    small_model = get_small_model_meta_load(reference_model).to(
        device="cuda", dtype=torch.bfloat16
    )
    with torch.device("cuda"):
        do_test_decode_consistency(small_model, reference_model, compile=True)


@skip_if_no_cuda
@pytest.mark.parametrize("mmap", [True, False])
def test_small_model_decode_compile_disk_load(mmap):
    reference_model = get_reference_model().to(device="cuda", dtype=torch.bfloat16)
    small_model = get_small_model_disk_load(reference_model, mmap=mmap).to(
        device="cuda", dtype=torch.bfloat16
    )
    with torch.device("cuda"):
        do_test_decode_consistency(small_model, reference_model, compile=True)


@skip_if_no_cuda
def test_small_model_decode_compile_load_model_method():
    reference_model = get_reference_model().to(device="cuda", dtype=torch.bfloat16)
    small_model = get_model_load_model_method(reference_model).to(
        device="cuda", dtype=torch.bfloat16
    )
    with torch.device("cuda"):
        do_test_decode_consistency(small_model, reference_model, compile=True)


@skip_if_no_cuda
def test_small_model_decode_no_compile_disk_load():
    reference_model = get_reference_model().to(device="cuda", dtype=torch.bfloat16)
    small_model = get_small_model_disk_load(reference_model).to(
        device="cuda", dtype=torch.bfloat16
    )
    with torch.device("cuda"):
        do_test_decode_consistency(small_model, reference_model, compile=False)


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


@skip_if_no_cuda
def test_llama_decode_load_model_method():
    with torch.device("cuda"):
        name = "unsloth/Llama-3.2-1B-Instruct"
        reference_model = LlamaForCausalLM.from_pretrained(
            name, torch_dtype=torch.bfloat16
        )
        config = ModelArgs.from_name(name)
        our_model = get_model_load_model_method(reference_model, config=config)
        do_test_decode_consistency(our_model, reference_model, compile=True)


@torch.inference_mode()
def do_test_decode_consistency(
    model: Transformer, reference_model: LlamaForCausalLM, compile=False
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

    input_ids = torch.zeros((batch_size, prompt_seq_length), dtype=torch.int)
    input_ids[0] = ref_output[0, :prompt_seq_length]
    for i in range(1, batch_size):
        input_ids[i, i:] = input_ids[0, :-i]
    start_inds = torch.arange(batch_size, dtype=torch.int)

    model.setup_caches(batch_size, max_seq_length)

    output = generate(
        model,
        input_ids=input_ids,
        start_inds=start_inds,
        max_seq_length=max_seq_length,
        max_new_tokens=max_seq_length - prompt_seq_length,
        device=input_ids.device,
        compile=compile,
        sampling=SamplingConfig(top_k=None, temperature=0.0),
    )

    print("REF OUT  ", ref_output[0].cpu())
    for b in range(batch_size):
        print(f"OUR OUT {b}", output[b, b:].cpu())
    assert (output[0].cpu() == ref_output[0].cpu()).all()
    for b in range(1, batch_size):
        assert (output[b, b:].cpu() == ref_output[0, :-b].cpu()).all(), (
            f"Output mismatch, at batch {b}"
        )
