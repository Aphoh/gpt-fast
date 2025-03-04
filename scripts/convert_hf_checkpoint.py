# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Optional
from safetensors.torch import load_file as load_safetensors_file
import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from model import ModelArgs


@torch.inference_mode()
def convert_hf_checkpoint(
    *,
    checkpoint_dir: Path = Path(
        "checkpoints/meta-Transformer/Transformer-2-7b-chat-hf"
    ),
    model_name: Optional[str] = None,
) -> None:
    if model_name is None:
        model_name = checkpoint_dir.name

    config = ModelArgs.from_name(model_name)
    print(f"Model config {config.__dict__}")

    # Load the json file containing weight mapping
    model_map_json_safetensors = checkpoint_dir / "model.safetensors.index.json"
    model_map_json_pytorch = checkpoint_dir / "pytorch_model.bin.index.json"
    model_map_json = None

    try:
        assert model_map_json_safetensors.is_file()
        model_map_json = model_map_json_safetensors
        print(f"Found safetensors index at {model_map_json_safetensors}")
    except AssertionError:
        print(f"{model_map_json_safetensors} not found")
    if model_map_json is None:
        try:
            assert model_map_json_pytorch.is_file()
            model_map_json = model_map_json_pytorch
            print(f"Found pytorch index at {model_map_json_pytorch}")
        except AssertionError:
            print(f"{model_map_json_pytorch} not found")

    if model_map_json is not None:
        with open(model_map_json) as json_map:
            bin_index = json.load(json_map)

        bin_files = {checkpoint_dir / bin for bin in bin_index["weight_map"].values()}
    else:
        bin_file = checkpoint_dir / "pytorch_model.bin"
        st_file = checkpoint_dir / "model.safetensors"
        if bin_file.is_file():
            print("Found pytorch_model.bin")
            bin_files = {bin_file}
        elif st_file.is_file():
            print("Found model.safetensors")
            bin_files = {st_file}
        else:
            raise FileNotFoundError(
                f"Could not find pytorch_model.bin or model.safetensors in {checkpoint_dir}"
            )

    def permute(w, n_head):
        dim = config.dim
        return (
            w.view(n_head, 2, config.head_dim // 2, dim)
            .transpose(1, 2)
            .reshape(config.head_dim * n_head, dim)
        )

    def permute_bias(b, n_head):
        return (
            b.view(n_head, 2, config.head_dim // 2)
            .transpose(1, 2)
            .reshape(config.head_dim * n_head)
        )

    merged_result = {}
    for file in sorted(bin_files):
        if "safetensors" in str(file):
            state_dict = load_safetensors_file(str(file), device="cpu")
            merged_result.update(state_dict)
        else:
            state_dict = torch.load(
                str(file), map_location="cpu", mmap=True, weights_only=True
            )
            merged_result.update(state_dict)

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

    for key, value in merged_result.items():
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
                q = permute(q, config.n_head)
                k = permute(k, config.n_local_heads)
            elif "bias" in key:
                q = permute_bias(q, config.n_head)
                k = permute_bias(k, config.n_local_heads)
            final_result[key.replace("wq", "wqkv")] = torch.cat([q, k, v])
            del final_result[key]
            del final_result[key.replace("wq", "wk")]
            del final_result[key.replace("wq", "wv")]

    for key in final_result.keys():
        print(f"{key}: {final_result[key].shape}")

    print(f"Saving checkpoint to {checkpoint_dir / 'model.pth'}")
    torch.save(final_result, checkpoint_dir / "model.pth")
    if "llama-3-" in model_name.lower() or "llama-3.1-" in model_name.lower():
        if "llama-3.1-405b" in model_name.lower():
            original_dir = checkpoint_dir / "original" / "mp16"
        else:
            original_dir = checkpoint_dir / "original"
        tokenizer_model = original_dir / "tokenizer.model"
        tokenizer_model_tiktoken = checkpoint_dir / "tokenizer.model"
        print(f"Copying {tokenizer_model} to {tokenizer_model_tiktoken}")
        shutil.copy(tokenizer_model, tokenizer_model_tiktoken)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert HuggingFace checkpoint.")
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=Path("checkpoints/meta-llama/llama-2-7b-chat-hf"),
    )
    parser.add_argument("--model_name", type=str, default=None)

    args = parser.parse_args()
    convert_hf_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        model_name=args.model_name,
    )
