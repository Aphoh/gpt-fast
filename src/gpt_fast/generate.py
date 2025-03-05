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

from gpt_fast.inputs import Batch, read_ids, read_input_batches, write_outputs
from gpt_fast.util import input_pos_from_start_inds, load_model
from gpt_fast.model import Transformer
from gpt_fast.tokenizer import detokenize_output_ids, get_tokenizer, tokenize_and_pad
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
    """
    Decode a single tokenn from the model given the mask
    x: [B, 1] input ids
    input_pos: [B, 1] input position
    block_mask: BlockMask the base mask for the generation

    Returns (next_token: [B] next token,
             next_prob: [B] probability of the next token)
    """
    B, _ = x.shape
    assert input_pos.shape[-1] == 1
    assert x.shape[-1] == 1
    block_index = input_pos // block_mask.BLOCK_SIZE[0]
    # TODO: check this is getting it across the batch correctly
    mask = block_mask[torch.arange(B, device=x.device), :, block_index]
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
    **sampling_kwargs,
) -> torch.IntTensor:
    """
    cur_token: [B] current token
    start_inds: [B] start index of the unpadded input in input_ids
    input_pos: [B] input position
    num_new_tokens: int number of new tokens to generate
    """
    block_mask = make_base_gen_mask(
        start_inds, model.max_seq_length, device=cur_token.device
    )
    new_tokens = []
    for i in range(num_new_tokens):
        next_token, next_prob = decode_one_token(
            model,
            cur_token.unsqueeze(-1),
            input_pos.unsqueeze(-1),
            block_mask,
            **sampling_kwargs,
        )
        input_pos += 1
        new_tokens.append(next_token.clone())
        cur_token = next_token.clone()

    return torch.cat(new_tokens, -1)


@torch.no_grad()
def generate(
    model: Transformer,
    input_ids: torch.Tensor,
    start_inds: torch.Tensor,
    max_seq_length: int,
    max_new_tokens: int,
    *,
    device: torch.device,
    **sampling_kwargs,
) -> torch.Tensor:
    """
    input_ids: [B, S] left padded batch of input ids
    start_inds: [B] start index of the unpadded input in input_ids
    max_seq_length: int maximum sequence length
    max_new_tokens: int maximum number of new tokens to generate

    Returns output_ids: [B, new_tokens] batch of generated tokens.
        Each output value ocurring after the stopping condition is hit has a value of -1.
    """
    # create an empty tensor of the expected final shape and fill in the current tokens
    B, S = input_ids.shape

    input_ids = input_ids.to(device=device)
    start_inds = start_inds.to(device=device)

    # This does mean the batching could cut stuff off if any prompt is too long, but let's just ignore that xD
    max_output_size = min(max_seq_length - S, max_new_tokens)
    output_ids = torch.empty(B, max_output_size, device=device, dtype=int)
    prefill_input_pos = input_pos_from_start_inds(start_inds, S, device=device)

    next_token = prefill(
        start_inds=start_inds,
        model=model,
        x=input_ids,
        input_pos=prefill_input_pos,
        **sampling_kwargs,
    ).clone()
    output_ids[:, 0] = next_token

    input_pos = prefill_input_pos[:, -1] + 1

    # TODO: stopping criteria
    generated_tokens = decode_n_tokens(
        model=model,
        cur_token=output_ids[:, 0],
        start_inds=start_inds,
        input_pos=input_pos,
        max_new_tokens=max_new_tokens - 1,
        **sampling_kwargs,
    )
    output_ids[:, 1 : 1 + generated_tokens.shape[-1]] = generated_tokens

    return output_ids


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
    max_seq_length: Optional[int],
    batch_size: int,
    top_k: int,
    temperature: float,
    checkpoint_path: Path,
    compile: bool,
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
    if compile:
        # TODO: do this cleaner
        global decode_one_token, prefill
        decode_one_token = torch.compile(
            decode_one_token, mode="reduce-overhead", fullgraph=True
        )
        prefill = torch.compile(prefill, fullgraph=True, dynamic=True)

    if max_seq_length is not None:
        assert max_seq_length <= model.max_seq_length, (
            f"{max_seq_length} > {model.max_seq_length}"
        )
    else:
        max_seq_length = model.max_seq_length

    with device:
        model.setup_caches(max_batch_size=batch_size, max_seq_length=max_seq_length)

    with open(output_file, "a") as f:
        for batch in tqdm(input_iterator, desc="Generating"):
            batch: Batch
            device_sync(device=device)
            encoded = tokenize_and_pad(batch.texts, tokenizer, max_seq_length)
            output = generate(
                model=model,
                input_ids=encoded.padded,
                start_inds=encoded.start_inds,
                max_seq_length=max_seq_length,
                max_new_tokens=max_new_tokens,
                device=device,
                precision=precision,
                # sampling kwargs
                temperature=temperature,
                top_k=top_k,
            )

            completions = detokenize_output_ids(output.cpu(), tokenizer)
            write_outputs(f, batch, completions)
            # TODO do I need a model.reset_caches()


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
        "--max_seq_length",
        type=int,
        help="Maximum sequence length for generation. Useful for limiting kv cache size.",
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
    parser.add_argument(
        "-n",
        "--no_compile",
        action="store_false",
        dest="compile",
        help="Skip compilation",
    )
    parser.add_argument(
        "--device", type=str, default=default_device, help="Device to use"
    )
    args = parser.parse_args()

    input_file: Path = args.input_file
    output_file: Path = args.output_file
    max_new_tokens: int = args.max_new_tokens
    max_seq_length: Optional[int] = args.max_seq_length
    batch_size: int = args.batch_size
    top_k: int = args.top_k
    temperature: float = args.temperature
    checkpoint_path: Path = args.checkpoint_path
    compile: bool = args.compile
    device: torch.device = torch.device(args.device)

    args = parser.parse_args()
    main(
        input_file,
        output_file,
        max_new_tokens,
        batch_size,
        top_k,
        temperature,
        checkpoint_path,
        compile,
        device,
    )
