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


def convert_full_ft_state_dict(
    config: ModelArgs, state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    final_result = {}
    for layer_idx in range(config.n_layer):
        from_prefix = f"model.layers.{layer_idx}"
        to_prefix = f"layers.{layer_idx}"
        for to_name, from_name in [
            ("attention.wq", "self_attn.q_proj"),
            ("attention.wk", "self_attn.k_proj"),
            ("attention.wv", "self_attn.v_proj"),
            ("attention.wo", "self_attn.o_proj"),
            ("feed_forward.w3", "mlp.gate_proj"),
            ("feed_forward.w2", "mlp.down_proj"),
            ("feed_forward.w1", "mlp.up_proj"),
            ("feed_forward.w2", "mlp.c_proj"),
            ("feed_forward.w1", "mlp.c_fc"),
            ("attention_norm", "input_layernorm"),
            ("ffn_norm", "post_attention_layernorm"),
        ]:
            for r in ["weight", "bias"]:
                from_key = f"{from_prefix}.{from_name}.{r}"
                to_key = f"{to_prefix}.{to_name}.{r}"
                if from_key in state_dict:
                    final_result[to_key] = state_dict.pop(from_key)

    # Remove extraneous keys
    state_dict.pop("router.weight", None)
    for k in list(state_dict.keys()):
        if "mlp.experts" in k:
            state_dict.pop(k)

    if "lm_head.weight" in state_dict:
        final_result["output.weight"] = state_dict.pop("lm_head.weight")
    if "lm_head.bias" in state_dict:
        final_result["output.bias"] = state_dict.pop("lm_head.bias")
    final_result["tok_embeddings.weight"] = state_dict.pop("model.embed_tokens.weight")
    final_result["norm.weight"] = state_dict.pop("model.norm.weight")
    if "model.norm.bias" in state_dict:
        final_result["norm.bias"] = state_dict.pop("model.norm.bias")
    assert len(state_dict) == 0, f"Found remaining keys{state_dict.keys()}"

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


# Compile should save some memory here...
@torch.compile
def _merge_lora_weight(
    base_weight: torch.Tensor, lora_a: torch.Tensor, lora_b: torch.Tensor
):
    # base_weight: [OUT, IN]
    # lora_a: [R, IN]
    # lora_b: [OUT, R]
    lora_expanded = lora_b @ lora_a
    return base_weight + lora_expanded


def _merge_concat_lora_qkv_weight(
    base_wqkv: torch.Tensor,
    lora_q_a: torch.Tensor,
    lora_q_b: torch.Tensor,
    lora_k_a: torch.Tensor,
    lora_k_b: torch.Tensor,
    lora_v_a: torch.Tensor,
    lora_v_b: torch.Tensor,
    n_head: int,
    n_local_heads: int,
    head_dim: int,
    dim: int,
) -> torch.Tensor:
    # Expand LoRA weights for q, k, v
    lora_q_expanded = lora_q_b @ lora_q_a
    lora_k_expanded = lora_k_b @ lora_k_a
    lora_v_expanded = lora_v_b @ lora_v_a

    # Permute expanded weights or biases
    lora_q_permuted = permute(lora_q_expanded, n_head, head_dim, dim)
    lora_k_permuted = permute(lora_k_expanded, n_local_heads, head_dim, dim)

    # Concatenate permuted Q, K and V
    lora_qkv_expanded = torch.cat([lora_q_permuted, lora_k_permuted, lora_v_expanded])

    # Merge with base wqkv weight
    return base_wqkv + lora_qkv_expanded


def merge_lora_state_dict(
    config: ModelArgs,
    full_state_dict: Dict[str, torch.Tensor],
    state_dict: Dict[str, torch.Tensor],
):
    final_result = dict(full_state_dict)
    for layer_idx in range(config.n_layer):
        from_prefix = f"model.layers.{layer_idx}"
        to_prefix = f"layers.{layer_idx}"
        for to_name, from_name in [
            ("attention.wo", "self_attn.o_proj"),
            ("feed_forward.w3", "mlp.gate_proj"),
            ("feed_forward.w2", "mlp.down_proj"),
            ("feed_forward.w1", "mlp.up_proj"),
            ("feed_forward.w2", "mlp.c_proj"),
            ("feed_forward.w1", "mlp.c_fc"),
        ]:
            to_key = f"{to_prefix}.{to_name}.weight"
            if to_key in full_state_dict:
                base_weight = full_state_dict[to_key]
                lora_a_key = f"{from_prefix}.{from_name}.low_rank_linear.lora_a.weight"
                lora_b_key = f"{from_prefix}.{from_name}.low_rank_linear.lora_b.weight"
                if lora_a_key in state_dict:
                    lora_a = state_dict.pop(lora_a_key)
                    lora_b = state_dict.pop(lora_b_key)
                    final_result[to_key] = _merge_lora_weight(
                        base_weight.to(lora_a.dtype), lora_a, lora_b
                    )

        wqkv_key = f"{to_prefix}.attention.wqkv.weight"
        attn_key = (
            from_prefix + ".self_attn.{wtype}_proj.low_rank_linear.lora_{ltype}.weight"
        )
        lora_q_a = state_dict.pop(attn_key.format(wtype="q", ltype="a"))
        lora_q_b = state_dict.pop(attn_key.format(wtype="q", ltype="b"))
        lora_k_a = state_dict.pop(attn_key.format(wtype="k", ltype="a"))
        lora_k_b = state_dict.pop(attn_key.format(wtype="k", ltype="b"))
        lora_v_a = state_dict.pop(attn_key.format(wtype="v", ltype="a"))
        lora_v_b = state_dict.pop(attn_key.format(wtype="v", ltype="b"))
        final_result[wqkv_key] = _merge_concat_lora_qkv_weight(
            base_wqkv=full_state_dict[wqkv_key].to(lora_q_a.dtype),
            lora_q_a=lora_q_a,
            lora_q_b=lora_q_b,
            lora_k_a=lora_k_a,
            lora_k_b=lora_k_b,
            lora_v_a=lora_v_a,
            lora_v_b=lora_v_b,
            n_head=config.n_head,
            n_local_heads=config.n_local_heads,
            head_dim=config.head_dim,
            dim=config.dim,
        )
    state_dict.pop("router.weight", None)
    assert len(state_dict) == 0, f"Found remaining keys{state_dict.keys()}"
    return final_result


def convert_routable_state_dict(
    config: ModelArgs, state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    rargs = config.routable_args
    if rargs.disable_expert_mask:
        raise ValueError("full ft baseline")

    if any("lora" in k for k in state_dict.keys()):
        raise ValueError("lora baseline")

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
