

import torch
from gpt_fast.generate import prefill
from gpt_fast.mask_utils import make_prefill_mask
from gpt_fast.model import Transformer, ModelArgs, RoutableArgs

def small_config(**kwargs):
    return ModelArgs(
        vocab_size=1000,
        dim=16,
        intermediate_size=32,
        n_head=4,
        n_local_heads=2,
        n_layer=6,
        block_size=1024,
        routable_args=RoutableArgs(
            num_experts=4,
            expert_rank=4,
            top_k=2,
            router_act_before_topk=True,
            **kwargs
        )
    )


def test_routed_populates_mask():
    SEQ_LEN = 32
    config = small_config()
    model = Transformer(config)
    model.setup_caches(1, SEQ_LEN)

    assert (model.expert_mask == 0.0).all()
    inputs = torch.randint(0, config.vocab_size, (1, SEQ_LEN))
    input_pos = torch.arange(SEQ_LEN).unsqueeze(0)
    seqlens = torch.randint(1, SEQ_LEN, (1,))
    prefill_mask = make_prefill_mask(seqlens=seqlens, query_len=SEQ_LEN, max_seqlen=SEQ_LEN, compile=False)
    prefill(
        model,
        x=inputs,
        input_pos=input_pos,
        prefill_mask=prefill_mask,
        seqlens=seqlens,
        compile=False
    )
    assert not (model.expert_mask == 0.0).all()

def test_routed_prefill_experts():
    SEQ_LEN = 32
    config = small_config(prefill_expert=True)
    model = Transformer(config)
    model.setup_caches(1, SEQ_LEN)

    assert (model.expert_mask == 0.0).all()
    inputs = torch.randint(0, config.vocab_size, (1, SEQ_LEN))
    input_pos = torch.arange(SEQ_LEN).unsqueeze(0)
    seqlens = torch.randint(1, SEQ_LEN, (1,))
    prefill_mask = make_prefill_mask(seqlens=seqlens, query_len=SEQ_LEN, max_seqlen=SEQ_LEN, compile=False)
    prefill(
        model,
        x=inputs,
        input_pos=input_pos,
        prefill_mask=prefill_mask,
        seqlens=seqlens,
        compile=False
    )
    assert not (model.expert_mask == 0.0).all()
    assert (model.expert_mask[:, :config.routable_args.prefill_expert_size] == 0.0).all()