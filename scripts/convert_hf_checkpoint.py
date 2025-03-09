# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import shutil
from pathlib import Path
from typing import Optional
from safetensors.torch import load_file as load_safetensors_file
import torch
from gpt_fast.model import ModelArgs
from gpt_fast.ckpt_utils import convert_state_dict


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

    final_result = convert_state_dict(config, merged_result)

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
