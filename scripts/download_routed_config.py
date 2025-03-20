"""
Given a Weights & Biases run ID and Google Cloud Storage path, this script downloads the safetensors file from GCS and pulls the model configuration from Weights & Biases.
Then, writes this to `checkpoints/base/model/path/name.json` and `checkpoints/base/model/path/name.safetensors`.
"""

import dataclasses
import json
from pathlib import Path
import fsspec
import wandb

from gpt_fast.ckpt_utils import RoutableCfg
from gpt_fast.model import RoutableArgs
import argparse


def download_safetensors_from_gcs(
    wandb_id: str, gcs_path: str, output_path: Path
) -> str:
    """
    Download a safetensors file from Google Cloud Storage using gcsfs.

    Args:
        wandb_id: The Weights & Biases run ID
        gcs_path: The Google Cloud Storage path prefix
        output_path: path to save the file
    """

    # Construct the full GCS path
    if not gcs_path.startswith("gs://"):
        gcs_path = f"gs://{gcs_path}"

    if not gcs_path.endswith("/"):
        gcs_path = f"{gcs_path}/"

    full_gcs_path = f"{gcs_path}{wandb_id}_state_dict.safetensors"

    # Use gcsfs to download the file
    gcs = fsspec.filesystem("gs")
    gcs.get(full_gcs_path, str(output_path))

    print(f"Downloaded {full_gcs_path} to {output_path}")
    return str(output_path)


def pull_config_from_wandb(
    api: wandb.Api, wandb_proj: str, wandb_id: str
) -> RoutableCfg:
    run = api.run(f"{wandb_proj}/{wandb_id}")
    mcfg = run.config["model"]
    routable_args = RoutableArgs(
        num_experts=mcfg["num_experts"],
        expert_rank=mcfg["expert_rank"],
        top_k=mcfg["top_k"],
        disable_expert_mask=mcfg["disable_expert_mask"],
        ident_expert_mask=mcfg["ident_expert_mask"],
        scale=mcfg["scale"],
        prefill_expert=mcfg["prefill_expert"],
        route_each_layer=mcfg["route_each_layer"],
        router_activation=mcfg["router_activation"],
        router_act_before_topk=mcfg["router_act_before_topk"],
    )
    middle_token = run.config["data"]["middle_token"]
    rconfig = RoutableCfg(
        args=routable_args,
        base_model=run.config["initialize_from_hf"],
        fim_middle_token=middle_token,
    )
    return rconfig


def main():
    parser = argparse.ArgumentParser(
        description="Download safetensors from GCS and pull config from wandb"
    )
    parser.add_argument(
        "-n", "--name", type=str, required=True, help="Human readable name"
    )
    parser.add_argument(
        "-i", "--wandb-id", type=str, required=True, help="Weights & Biases run ID"
    )
    parser.add_argument(
        "-p",
        "--wandb-proj",
        default="levanter-seqmoe",
        type=str,
        help="Weights & Biases run project",
    )
    parser.add_argument(
        "--gcs-path", type=str, required=True, help="Google Cloud Storage path prefix"
    )

    args = parser.parse_args()

    # Pull config from wandb
    api = wandb.Api()
    config = pull_config_from_wandb(api, args.wandb_proj, args.wandb_id)
    cfg_dict = dataclasses.asdict(config)

    out_dir = Path("checkpoints") / config.base_model
    assert out_dir.is_dir(), f"Output directory {out_dir} does not exist"

    download_safetensors_from_gcs(
        wandb_id=args.wandb_id,
        gcs_path=args.gcs_path,
        output_path=out_dir / f"{args.name}.safetensors",
    )

    with open(out_dir / f"{args.name}.json", "w") as f:
        json.dump(cfg_dict, f)

    print(
        f"Successfully downloaded safetensors and wrote config to {out_dir / args.name}.json"
    )


if __name__ == "__main__":
    main()
