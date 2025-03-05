# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch._dynamo.config
import torch._inductor.config
from torch.nn.attention.flex_attention import BlockMask
import tqdm

from gpt_fast.inputs import Batch, read_ids, read_input_batches
from gpt_fast.util import load_model
from gpt_fast.model import Transformer
from gpt_fast.tokenizer import get_tokenizer, tokenize_and_pad
from gpt_fast.mask_utils import make_base_gen_mask, make_prefill_mask


def device_sync(device: str) -> None:
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif ("cpu" in device) or ("mps" in device):
        pass
    else:
        print(f"device={device} is not yet suppported")


default_device: str = "cuda" if torch.cuda.is_available() else "cpu"


def multinomial_sample_one_no_sync(
    probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[:, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def prefill(
    start_inds: torch.Tensor,
    model: Transformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    **sampling_kwargs,
) -> torch.Tensor:
    # start_inds: [B]
    # input_pos: [B, S]
    mask = make_prefill_mask(
        start_inds, input_pos, device=x.device, max_seq_length=model.max_seq_length
    )
    logits = model(mask, x, input_pos)
    return sample(logits, **sampling_kwargs)[0]


def decode_one_token(
    model: Transformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    block_mask: BlockMask,
    **sampling_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    block_index = input_pos // block_mask.BLOCK_SIZE[0]
    mask = block_mask[:, :, block_index]
    mask.mask_mod = block_mask.mask_mod
    mask.seq_lengths = (1, model.max_seq_length)
    logits = model(mask, x, input_pos)
    return sample(logits, **sampling_kwargs)


def decode_n_tokens(
    model: Transformer,
    cur_token: torch.Tensor,
    start_inds: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    callback=lambda _: _,
    **sampling_kwargs,
):
    block_mask = make_base_gen_mask(
        start_inds, model.max_seq_length, device=cur_token.device
    )
    new_tokens, new_probs = [], []
    for i in range(num_new_tokens):
        next_token, next_prob = decode_one_token(
            model, cur_token, input_pos, block_mask, **sampling_kwargs
        )
        input_pos += 1
        new_tokens.append(next_token.clone())
        callback(new_tokens[-1])
        new_probs.append(next_prob.clone())
        cur_token = next_token.clone()

    return new_tokens, new_probs


def model_forward(model, x, input_pos):
    return model(x, input_pos)


@torch.no_grad()
def generate(
    model: Transformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    batch_size: int,
    **sampling_kwargs,
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """
    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(-1)
    T_new = T + max_new_tokens
    max_seq_length = min(T_new, model.config.block_size)

    device, dtype = prompt.device, prompt.dtype
    with torch.device(device):
        model.setup_caches(max_batch_size=batch_size, max_seq_length=max_seq_length)

    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(batch_size, T_new, dtype=dtype, device=device)
    # We are just making the same prompt for every batch
    prompt = prompt.view(1, -1).repeat(batch_size, 1)
    empty[:, :T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device)

    start_inds = torch.tensor([0], device=device)
    next_token = prefill(
        start_inds, model, prompt.view(batch_size, -1), input_pos, **sampling_kwargs
    ).clone()
    seq[:, T] = next_token.squeeze()

    input_pos = torch.tensor([T], device=device, dtype=torch.int)

    generated_tokens, _ = decode_n_tokens(
        model,
        next_token.view(batch_size, -1),
        start_inds,
        input_pos,
        max_new_tokens - 1,
        **sampling_kwargs,
    )
    seq[:, T + 1 :] = torch.cat(generated_tokens, dim=-1)

    generate_stats = {}
    return seq, generate_stats


def _load_model(checkpoint_path, device, precision, use_tp):
    with torch.device("meta"):
        model = Transformer.from_name(checkpoint_path.parent.name)

    model = load_model(model, checkpoint_path, precision=precision, device=device)

    if use_tp:
        from gpt_fast.tp import apply_tp

        print("Applying tensor parallel to model ...")
        apply_tp(model)

    model = model.to(device=device, dtype=precision)
    return model.eval()


def main(
    input_file: Path,
    output_file: Path,
    max_new_tokens: int,
    batch_size: int,
    top_k: int,
    temperature: float,
    checkpoint_path: Path,
    profile: Optional[Path],
    device: torch.device,
) -> None:
    """Generates text samples based on a pre-trained Transformer model and tokenizer."""
    ### Preamble
    assert checkpoint_path.is_file(), checkpoint_path
    assert input_file.is_file(), output_file

    global print
    from gpt_fast.tp import maybe_init_dist

    rank = maybe_init_dist()
    use_tp = rank is not None
    if use_tp:
        if rank != 0:
            # only print on rank 0
            print = lambda *args, **kwargs: None  # noqa: E731

    print(f"Using device={device}")
    precision = torch.bfloat16

    completed_ids = set()
    if output_file.is_file():
        completed_ids = read_ids(output_file)
        print(f"Skipping {len(completed_ids)} completed ids:")

    input_iterator = read_input_batches(input_file, batch_size, completed_ids)

    print("Loading model ...")
    t0 = time.time()
    model = _load_model(checkpoint_path, device, precision, use_tp)

    device_sync(device=device)  # MKG
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    tokenizer = get_tokenizer(checkpoint_path.parent)

    torch.manual_seed(1234)
    global decode_one_token, prefill
    decode_one_token = torch.compile(
        decode_one_token, mode="reduce-overhead", fullgraph=True
    )

    # Uncomment to squeeze more perf out of prefill
    prefill = torch.compile(prefill, fullgraph=True, dynamic=True)

    for batch in tqdm(input_iterator, desc="Generating"):
        batch: Batch
        device_sync(device=device)
        batch_encoded = tokenize_and_pad(batch.texts, tokenizer, model.max_seq_length)
        output, metrics = generate(
            model,
            batch_encoded.padded,
            max_new_tokens,
            batch_size,
            interactive=False,
            temperature=temperature,
            top_k=top_k,
        )
        # TODO: Write output to file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Your CLI description.")

    parser.add_argument(
        "-i",
        "--input_file",
        type=Path,
        help="Input jsonl with 'prompt' and 'id' keys",
        default=Path("sample_input.jsonl"),
    )
    parser.add_argument("-o", "--output_file", type=Path, help="Output jsonl path")
    parser.add_argument(
        "--max_new_tokens", type=int, default=200, help="Maximum number of new tokens."
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size to run with"
    )
    parser.add_argument("--top_k", type=int, default=200, help="Top-k for sampling.")
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Temperature for sampling."
    )
    parser.add_argument(
        "-c",
        "--checkpoint_path",
        type=Path,
        default=Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"),
        help="Model checkpoint path.",
    )
    parser.add_argument("--profile", type=Path, default=None, help="Profile path.")
    parser.add_argument(
        "--device", type=str, default=default_device, help="Device to use"
    )
    args = parser.parse_args()

    input_file: Path = args.input_file
    output_file: Path = args.output_file
    max_new_tokens: int = args.max_new_tokens
    batch_size: int = args.batch_size
    top_k: int = args.top_k
    temperature: float = args.temperature
    checkpoint_path: Path = args.checkpoint_path
    profile: Optional[Path] = args.profile
    device: torch.device = torch.device(args.device)

    args = parser.parse_args()
    main(
        input_file,
        args.output_file,
        args.max_new_tokens,
        args.batch_size,
        args.top_k,
        args.temperature,
        args.checkpoint_path,
        args.profile,
        args.device,
    )
