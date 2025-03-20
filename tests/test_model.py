from gpt_fast.generate import decode_one_token, prefill
from gpt_fast.mask_utils import get_gen_mask, make_prefill_mask
import torch
import pytest
from transformers import LlamaForCausalLM, LlamaConfig, AutoConfig
from gpt_fast.ckpt_utils import convert_state_dict
from gpt_fast.model import Transformer, ModelArgs
from torch.testing import assert_close


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
        rope_base=config.rope_theta,
        norm_eps=config.rms_norm_eps,
    )

    model = LlamaForCausalLM(config).to(torch.float)
    state_dict = model.state_dict()

    converted = convert_state_dict(model_args, state_dict)
    tformer = Transformer(model_args).to(torch.float)
    tformer.load_state_dict(converted)

    consistency(model, tformer)


def test_llama_1b_consistent():
    if not torch.cuda.is_available():
        pytest.skip("cuda not found")
    with torch.no_grad():
        with torch.device("cuda") as device:
            name = "unsloth/Llama-3.2-1B-Instruct"
            config = AutoConfig.from_pretrained(name)
            model = LlamaForCausalLM(config).to(device=device, dtype=torch.float)
            state_dict = model.state_dict()

            model_args = ModelArgs.from_name(name)
            converted = convert_state_dict(model_args, state_dict)
            tformer = Transformer(model_args).to(device=device, dtype=torch.float)
            tformer.load_state_dict(converted)

            consistency(model, tformer, atol=3e-5, rtol=1e-2)


@torch.no_grad()
def consistency(ref_model, tformer, **kwargs):
    check_prefill_consistent(ref_model, tformer, **kwargs)
    check_decode_consistent(tformer, **kwargs)


def setup_test_inputs(gen_len=40, prefill_len=32, batch_size=4):
    input_ids = torch.randint(0, 1000, (batch_size, gen_len))
    seqlens = torch.randint(4, prefill_len - 4, (batch_size,))

    input_pos = torch.arange(gen_len).unsqueeze(0).expand(batch_size, -1)

    prefill_input_ids = input_ids[:, :prefill_len].clone()
    prefill_input_pos = input_pos[:, :prefill_len].clone()
    for i in range(batch_size):
        prefill_input_ids[i, seqlens[i] :] = 0

    return (
        input_ids,
        seqlens,
        input_pos,
        prefill_input_ids,
        prefill_input_pos,
    )


def check_prefill_consistent(ref_model, tformer, **kwargs):
    """Test that prefill outputs match between reference model and our Transformer"""
    input_ids, seqlens, _, prefill_input_ids, prefill_input_pos = setup_test_inputs()
    B, GEN_S = input_ids.shape

    attn_mask = torch.ones_like(prefill_input_ids)
    ref_output = ref_model(input_ids=prefill_input_ids, attention_mask=attn_mask).logits

    tformer.setup_caches(B, GEN_S)
    prefill_mask = make_prefill_mask(
        seqlens, prefill_input_ids.shape[1], GEN_S, compile=False
    )
    our_output = prefill(
        tformer,
        x=prefill_input_ids,
        input_pos=prefill_input_pos,
        prefill_mask=prefill_mask,
        seqlens=seqlens,
        return_logits=True,
        compile=False,
    )

    for b in range(B):
        assert_close(
            ref_output[b, : seqlens[b]],
            our_output[b, : seqlens[b]],
            **kwargs,
        )


def check_decode_consistent(tformer: Transformer, **kwargs):
    """Test that single token decode matches prefill outputs"""
    input_ids, seqlens, input_pos, prefill_input_ids, prefill_input_pos = (
        setup_test_inputs()
    )
    B, GEN_S = input_ids.shape
    _, PREFILL_S = prefill_input_ids.shape

    tformer.setup_caches(B, GEN_S)

    full_seqlens = torch.ones(B, dtype=int) * GEN_S
    full_prefill_mask = make_prefill_mask(full_seqlens, GEN_S, GEN_S, compile=False)
    full_output_logits = prefill(
        tformer,
        x=input_ids,  # Use full sequence for prefill reference
        input_pos=input_pos,
        seqlens=full_seqlens,
        prefill_mask=full_prefill_mask,
        return_logits=True,
        compile=False,
    )

    # Clear caches
    tformer.clear_caches()
    tformer.setup_caches(B, GEN_S)
    # Prefill the sequence to populate kv caches
    prefill_mask = make_prefill_mask(
        seqlens, prefill_input_ids.shape[1], GEN_S, compile=False
    )
    prefill(
        tformer,
        x=prefill_input_ids,  # Use full sequence for prefill reference
        input_pos=prefill_input_pos,
        prefill_mask=prefill_mask,
        seqlens=seqlens,
        compile=False,
    )

    # Now test individual token decoding against prefill outputs
    offsets = seqlens.clone()
    for i in range(PREFILL_S, GEN_S):
        gen_mask_i = get_gen_mask(offsets=offsets, max_seqlen=GEN_S, BLOCK_SIZE=8)
        b_inds = torch.arange(B)
        cur_token = input_ids[b_inds, offsets]
        next_token_logits = decode_one_token(
            gen_mask_i=gen_mask_i,
            model=tformer,
            cur_token=cur_token,
            input_pos=offsets,
            compile=False,
            return_logits=True,
        )
        for b in range(B):
            idx = offsets[b]
            assert_close(
                full_output_logits[b, idx],
                next_token_logits[b].squeeze(0),
                **kwargs,
            )
        offsets += 1
