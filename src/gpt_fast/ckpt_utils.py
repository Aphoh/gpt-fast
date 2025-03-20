import dataclasses
import re
from typing import Dict
from .model import ModelArgs, RoutableArgs
import torch


@dataclasses.dataclass
class RoutableCfg:
    args: RoutableArgs
    base_model: str
    fim_middle_token: str


def permute(w, n_head, head_dim, dim):
    return (
        w.view(n_head, 2, head_dim // 2, dim)
        .transpose(1, 2)
        .reshape(head_dim * n_head, dim)
    )


def permute_bias(b, n_head, head_dim):
    return b.view(n_head, 2, head_dim // 2).transpose(1, 2).reshape(head_dim * n_head)


def convert_routable_state_dict(
    config: ModelArgs, state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    rargs = config.routable_args
    if rargs.disable_expert_mask:
        raise ValueError("TODO: full ft baseline")

    if any("lora" in k for k in state_dict.keys()):
        raise ValueError("TODO: lora")

    final_result = {}
    for layer_idx in range(config.n_layer):
        from_prefix = f"model.layers.{layer_idx}.mlp.experts"
        to_prefix = f"layers.{layer_idx}.routed_experts"
        final_result[f"{to_prefix}.w1.weight"] = state_dict[
            f"{from_prefix}.up_proj.weight"
        ]
        final_result[f"{to_prefix}.w2.weight"] = state_dict[
            f"{from_prefix}.down_proj.weight"
        ]
        if config.glu:
            final_result[f"{to_prefix}.w3.weight"] = state_dict[
                f"{from_prefix}.gate_proj.weight"
            ]
    if rargs.prefill_expert:
        topk = rargs.top_k
        r_weight = state_dict["router.weight"]
        final_result["router.weight"] = r_weight[topk:]  # remove prefill expert weights
    else:
        final_result["router.weight"] = state_dict["router.weight"]
    return final_result


def convert_state_dict(config: ModelArgs, state_dict: Dict[str, torch.Tensor]):
    final_result = {}
    weight_map = {
        "model.embed_tokens": "tok_embeddings",
        "model.layers.{}.self_attn.q_proj": "layers.{}.attention.wq",
        "model.layers.{}.self_attn.k_proj": "layers.{}.attention.wk",
        "model.layers.{}.self_attn.v_proj": "layers.{}.attention.wv",
        "model.layers.{}.self_attn.o_proj": "layers.{}.attention.wo",
        "model.layers.{}.mlp.gate_proj": "layers.{}.feed_forward.w1",
        "model.layers.{}.mlp.up_proj": "layers.{}.feed_forward.w3",
        "model.layers.{}.mlp.down_proj": "layers.{}.feed_forward.w2",
        "model.layers.{}.input_layernorm": "layers.{}.attention_norm",
        "model.layers.{}.post_attention_layernorm": "layers.{}.ffn_norm",
        "model.norm": "norm",
        "lm_head": "output",
        # Starcoder2
        "model.layers.{}.mlp.c_fc": "layers.{}.feed_forward.w1",
        "model.layers.{}.mlp.c_proj": "layers.{}.feed_forward.w2",
    }
    weight_map = {
        k + postfix: v + postfix
        for k, v in weight_map.items()
        for postfix in [".weight", ".bias"]
    }
    weight_map |= {
        "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
    }

    for key, value in state_dict.items():
        if "layers" in key:
            abstract_key = re.sub(r"(\d+)", "{}", key)
            layer_num = re.search(r"\d+", key).group(0)
            new_key = weight_map[abstract_key]
            if new_key is None:
                continue
            new_key = new_key.format(layer_num)
        else:
            new_key = weight_map[key]

        final_result[new_key] = value

    for key in tuple(final_result.keys()):
        if "wq" in key:
            q = final_result[key]
            k = final_result[key.replace("wq", "wk")]
            v = final_result[key.replace("wq", "wv")]
            if "weight" in key:
                q = permute(q, config.n_head, config.head_dim, config.dim)
                k = permute(k, config.n_local_heads, config.head_dim, config.dim)
            elif "bias" in key:
                q = permute_bias(q, config.n_head, config.head_dim)
                k = permute_bias(k, config.n_local_heads, config.head_dim)
            final_result[key.replace("wq", "wqkv")] = torch.cat([q, k, v])
            del final_result[key]
            del final_result[key.replace("wq", "wk")]
            del final_result[key.replace("wq", "wv")]

    return final_result
