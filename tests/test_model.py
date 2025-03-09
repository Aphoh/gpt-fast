import torch
from transformers import LlamaForCausalLM, AutoConfig
from gpt_fast.ckpt_utils import convert_state_dict
from gpt_fast.mask_utils import make_prefill_mask
from gpt_fast.model import Transformer, ModelArgs


def test_stories_consistent():
    name = "Xenova/llama2.c-stories15M"
    config = AutoConfig.from_pretrained(name)
    model = LlamaForCausalLM(config)
    state_dict = model.state_dict()

    model_args = ModelArgs.from_name(name)
    converted = convert_state_dict(model_args, state_dict)
    tformer = Transformer(model_args)
    tformer.load_state_dict(converted)

    S = 32
    input_ids = torch.arange(S).unsqueeze(0)
    attn_mask = torch.ones(S)

    ref_output = model(input_ids=input_ids, attention_mask=attn_mask).logits

    start_inds = torch.zeros(1, dtype=int)
    mask = make_prefill_mask(start_inds, S, S, device="cpu")
    input_pos = torch.arange(S).unsqueeze(0)
    tformer.setup_caches(1, S)
    output = tformer(mask=mask, idx=input_ids, input_pos=input_pos, start_inds=start_inds, offset=[0])

    assert torch.allclose(ref_output, output, atol=1e-4)
