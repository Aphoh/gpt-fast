from gpt_fast.generate import decode_one_token, prefill
from gpt_fast.mask_utils import get_gen_submask, make_base_mask
from gpt_fast.util import input_pos_from_start_inds
import torch
import pytest
from transformers import LlamaForCausalLM, LlamaConfig, AutoConfig
from gpt_fast.ckpt_utils import convert_state_dict
from gpt_fast.model import Transformer, ModelArgs


def test_small_model_consistent():
    torch.random.manual_seed(42)
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

    model = LlamaForCausalLM(config).to(torch.float16)
    state_dict = model.state_dict()

    converted = convert_state_dict(model_args, state_dict)
    tformer = Transformer(model_args).to(torch.float16)
    tformer.load_state_dict(converted)

    consistency(model, tformer)


def test_llama_1b_consistent():
    if not torch.cuda.is_available():
        pytest.skip("cuda not found")
    with torch.no_grad():
        with torch.device("cuda") as device:
            name = "unsloth/Llama-3.2-1B-Instruct"
            config = AutoConfig.from_pretrained(name)
            model = LlamaForCausalLM(config).to(device)
            state_dict = model.state_dict()

            model_args = ModelArgs.from_name(name)
            converted = convert_state_dict(model_args, state_dict)
            tformer = Transformer(model_args)
            tformer.load_state_dict(converted)

            consistency(model, tformer)


def setup_test_inputs(gen_len=40, prefill_len=32, batch_size=4):
    base_input_ids = torch.arange(gen_len, dtype=int).unsqueeze(0)
    input_ids = torch.zeros(batch_size, gen_len, dtype=int)
    input_ids[0] = base_input_ids[0]
    for i in range(1, batch_size):
        input_ids[i, i:] = input_ids[0, :-i]
    prefill_input_ids = input_ids[:, :prefill_len]

    start_inds = torch.arange(batch_size)
    input_pos = input_pos_from_start_inds(start_inds, gen_len)
    prefill_input_pos = input_pos_from_start_inds(start_inds, prefill_len)

    return (
        base_input_ids,
        input_ids,
        prefill_input_ids,
        start_inds,
        input_pos,
        prefill_input_pos,
    )


def check_prefill_consistent(ref_model, tformer):
    """Test that prefill outputs match between reference model and our Transformer"""
    gen_len = 40
    prefill_len = 32
    batch_size = 4

    base_input_ids, _, prefill_input_ids, start_inds, _, prefill_input_pos = (
        setup_test_inputs(gen_len, prefill_len, batch_size)
    )

    ref_output = ref_model(input_ids=base_input_ids[:, :prefill_len]).logits
    tformer.setup_caches(batch_size, gen_len)

    our_output = prefill(
        tformer,
        x=prefill_input_ids,
        start_inds=start_inds,
        input_pos=prefill_input_pos,
        max_seq_length=gen_len,
        return_logits=True,
        compile=False,
    )

    for i in range(batch_size):
        print(
            (ref_output[0, : prefill_len - i] - our_output[i, i:prefill_len])
            .abs()
            .max(dim=-1)
        )
        for s in range(prefill_len - i):
            assert torch.allclose(
                ref_output[0, s], our_output[i, i + s], atol=1e-1, rtol=1e-2
            ), f"Failed for batch index {i}, seq index {s}"


def check_decode_consistent(tformer: Transformer):
    """Test that single token decode matches prefill outputs"""
    gen_len = 40
    prefill_len = 32
    batch_size = 4

    _, input_ids, prefill_input_ids, start_inds, input_pos, prefill_input_pos = (
        setup_test_inputs(gen_len, prefill_len, batch_size)
    )

    tformer.setup_caches(batch_size, gen_len)

    prefill_output = prefill(
        tformer,
        x=input_ids,  # Use full sequence for prefill reference
        start_inds=start_inds,
        input_pos=input_pos,
        max_seq_length=gen_len,
        return_logits=True,
        compile=False,
    )

    # Clear caches
    tformer.clear_caches()
    tformer.setup_caches(batch_size, gen_len)
    # Prefill the sequence to populate kv caches
    prefill(
        tformer,
        x=prefill_input_ids,  # Use full sequence for prefill reference
        start_inds=start_inds,
        input_pos=prefill_input_pos,
        max_seq_length=gen_len,
        compile=False,
    )

    # Now test individual token decoding against prefill outputs
    device = input_ids.device
    base_mask = make_base_mask(
        batch_size, gen_len, gen_len, device=device, compile=False
    )
    for i in range(prefill_len, gen_len):
        gen_mask_i = get_gen_submask(base_mask, i)
        cur_token = input_ids[:, i]
        input_pos_gen = input_pos[:, i]
        next_token_logits = decode_one_token(
            gen_mask_i,
            tformer,
            cur_token,
            start_inds,
            input_pos_gen,
            offset=torch.tensor([i], device=device),
            compile=False,
            return_logits=True,
        )
        for b in range(batch_size):
            diff = prefill_output[b, i] - next_token_logits[b]
            assert torch.allclose(
                prefill_output[b, i], next_token_logits[b], atol=1e-3, rtol=1e-3
            ), (
                f"Failed for batch index {b}, seq index {i}, max diff: {diff.abs().max()}"
            )


@torch.no_grad()
def consistency(ref_model, tformer):
    check_prefill_consistent(ref_model, tformer)
    check_decode_consistent(tformer)
