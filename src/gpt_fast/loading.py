import dataclasses
import json
from typing import Optional

import torch
from gpt_fast.ckpt_utils import RoutableCfg
from gpt_fast.model import ModelArgs, RoutableArgs, Transformer
from gpt_fast.ckpt_utils import (
    convert_routable_state_dict,
    convert_full_ft_state_dict,
    merge_lora_state_dict,
)
from safetensors.torch import load_file as safetensors_load_file


def load_model(
    checkpoint_path, routable_path, device, precision
) -> tuple[Transformer, Optional[RoutableCfg]]:
    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)

    args: ModelArgs
    rcfg: Optional[RoutableCfg] = None
    if routable_path is not None:
        routable_ckpt_path = routable_path.with_suffix(".safetensors")
        rstate_dict = safetensors_load_file(str(routable_ckpt_path))
        with open(routable_path) as f:
            config_dict = json.load(f)
        rargs = RoutableArgs(**config_dict.pop("args"))
        rcfg = RoutableCfg(args=rargs, **config_dict)
        args = ModelArgs.from_name(rcfg.base_model)

        is_lora = any("lora" in k for k in rstate_dict.keys())
        is_full_ft = any("embed_tokens" in k for k in rstate_dict.keys())
        if is_lora:
            checkpoint = merge_lora_state_dict(args, checkpoint, rstate_dict)
            rcfg, rargs = None, None
        elif is_full_ft:
            checkpoint = convert_full_ft_state_dict(args, rstate_dict)
            rcfg, rargs = None, None
        else:
            args = dataclasses.replace(args, routable_args=rargs)
            checkpoint |= convert_routable_state_dict(args, rstate_dict)
    else:
        args = ModelArgs.from_name(checkpoint_path.parent.name)

    with torch.device("meta"):
        model = Transformer(args)

    if model.config.tie_embedding_weights:
        checkpoint["output.weight"] = checkpoint["tok_embeddings.weight"]

    model.load_state_dict(checkpoint, assign=True)
    model = model.to(dtype=precision, device=device)
    return model, rcfg
