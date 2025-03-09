from gpt_fast.generate import prefill
from gpt_fast.util import input_pos_from_start_inds
import torch
import pytest
from transformers import LlamaForCausalLM, AutoConfig
from gpt_fast.ckpt_utils import convert_state_dict
from gpt_fast.model import Transformer, ModelArgs


def test_small_model_consistent():
    name = "Xenova/llama2.c-stories15M"
    config = AutoConfig.from_pretrained(name)
    config.hidden_size = 256
    config.num_attention_heads = 8
    config.num_key_value_heads = 8
    config.head_dim = 256 // 8
    model = LlamaForCausalLM(config)
    state_dict = model.state_dict()

    model_args = ModelArgs.from_name(name)
    model_args.head_dim = config.head_dim
    model_args.dim = config.hidden_size
    model_args.n_head = config.num_attention_heads
    model_args.n_local_heads = config.num_key_value_heads
    converted = convert_state_dict(model_args, state_dict)
    tformer = Transformer(model_args)
    tformer.load_state_dict(converted)

    prefill_consistency(model, tformer)


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

            prefill_consistency(model, tformer)


def prefill_consistency(ref_model, tformer):
    S = 32
    input_ids = torch.ones(S, dtype=int).unsqueeze(0)
    attn_mask = torch.ones(S, dtype=int)

    ref_output = ref_model(input_ids=input_ids, attention_mask=attn_mask).logits

    B = 4
    tformer.setup_caches(4, S)

    input_ids = torch.ones(B, S, dtype=int)
    for i in range(B):
        input_ids[i, :i] = 0
    start_inds = torch.arange(B)
    input_pos = input_pos_from_start_inds(start_inds, S)

    prefill_logits = prefill(
        tformer,
        x=input_ids,
        start_inds=start_inds,
        input_pos=input_pos,
        max_seq_length=S,
        return_logits=True,
        compile=False,
    )

    for i in range(B):
        # In batch i, tokens i:S should be the same as 0:S-i in the reference
        assert torch.allclose(
            ref_output[0, 0 : S - i], prefill_logits[i, i:S], atol=1e-4
        ), f"Failed for bactch index {i}"
