# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from functools import partial
import time
from pathlib import Path
from typing import Optional

import torch
import torch._dynamo.config
import torch._inductor.config
from torch.nn.attention.flex_attention import BlockMask
from tqdm import tqdm

from gpt_fast.inputs import Batch, read_ids, read_input_batches, write_outputs
from gpt_fast.stopping import (
    StoppingCondition,
    combine_stopping_conditions,
    contains_word_stopping_condition,
    ignore_batch_inds,
)
from gpt_fast.util import load_model, maybe_compile
from gpt_fast.model import Transformer
from gpt_fast.tokenizer import detokenize_output_ids, get_tokenizer, tokenize_and_pad
from gpt_fast.mask_utils import (
    make_prefill_mask,
    get_gen_mask,
)


def device_sync(device: torch.device) -> None:
    if "cuda" in device.type:
        torch.cuda.synchronize(device)
    elif ("cpu" in device.type) or ("mps" in device.type):
        pass
    else:
        print(f"device={device} is not yet suppported")


default_device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class SamplingConfig:
    top_k: Optional[int] = None
    temperature: float = 1.0


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


def sample(logits, config: SamplingConfig):
    probs = logits_to_probs(logits[:, -1], config.temperature, config.top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


@maybe_compile(fullgraph=True, dynamic=False)
def prefill(
    model: Transformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    prefill_mask: BlockMask,
    seqlens: torch.Tensor,
    sampling: SamplingConfig = SamplingConfig(),
    return_logits=False,
) -> torch.Tensor:
    # x: [B, S]
    # input_pos: [B, S]
    logits = model(prefill_mask, x, input_pos)
    if return_logits:
        return logits
    b_inds = torch.arange(x.shape[0], device=x.device)
    # if seqlens is 0, we sample from the first token
    sample_idxs = (seqlens - 1).clamp(min=0)
    to_sample = logits[b_inds, sample_idxs].unsqueeze(1)
    return sample(to_sample, sampling)[0]


@maybe_compile(mode="reduce-overhead", fullgraph=False, dynamic=False)
def decode_one_token(
    gen_mask_i: BlockMask,
    model: Transformer,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    sampling: SamplingConfig = SamplingConfig(),
    return_logits=False,
) -> torch.IntTensor:
    logits = model(
        gen_mask_i,
        cur_token.unsqueeze(-1),
        input_pos=input_pos.unsqueeze(-1),
    )
    if return_logits:
        return logits
    next_token, _ = sample(logits, sampling)
    return next_token


def decode_n_tokens(
    model: Transformer,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    max_new_tokens: int,
    max_seqlen: int,
    sampling: SamplingConfig = SamplingConfig(),
    compile: bool = False,
    stopping_condition: Optional[StoppingCondition] = None,
) -> torch.IntTensor:
    """
    cur_token: [B] current token
    input_pos: [B] input position
    num_new_tokens: int number of new tokens to generate
    """
    assert cur_token.shape == input_pos.shape
    new_tokens = []
    stop_map = torch.zeros(cur_token.shape, dtype=torch.bool, device=cur_token.device)
    for i in (pbar := tqdm(range(max_new_tokens), desc="generating", leave=False)):
        gen_mask_i = get_gen_mask(input_pos, max_seqlen)
        next_token = decode_one_token(
            gen_mask_i=gen_mask_i,
            model=model,
            cur_token=cur_token,
            input_pos=input_pos,
            sampling=sampling,
            compile=compile,
        )
        input_pos += 1
        next_token[stop_map, 0] = -1
        new_tokens.append(next_token.clone())
        cur_token = next_token[:, 0].clone()
        if stopping_condition is not None:
            stop_map |= stopping_condition(torch.hstack(new_tokens))
            pbar.set_description(f"generating {stop_map.sum().item()}/{len(stop_map)}")
            if stop_map.all():
                break

    return torch.cat(new_tokens, -1)


@torch.no_grad()
def generate(
    model: Transformer,
    input_ids: torch.Tensor,
    seqlens: torch.Tensor,
    max_seqlen: int,
    max_new_tokens: int,
    *,
    device: torch.device,
    compile: bool,
    sampling: SamplingConfig = SamplingConfig(),
    stopping_condition: Optional[StoppingCondition] = None,
) -> torch.Tensor:
    """
    input_ids: [B, S] left padded batch of input ids
    seqlens: [B] end index of the unpadded input of each batch
    max_seqlen: int maximum sequence length
    max_new_tokens: int maximum number of new tokens to generate

    Returns output_ids: [B, new_tokens] batch of generated tokens.
        Each output value ocurring after the stopping condition is hit has a value of -1.
    """
    # create an empty tensor of the expected final shape and fill in the current tokens
    B, S = input_ids.shape

    input_ids = input_ids.to(device=device, dtype=torch.int).clamp(min=0)
    seqlens = seqlens.to(device=device, dtype=torch.int)

    # This does mean the batching could cut stuff off if any prompt is too long, but let's just ignore that xD
    if S + max_new_tokens > max_seqlen:
        raise ValueError(
            f"Input sequence length {S} + max_new_tokens {max_new_tokens} >= max_seqlen {max_seqlen}"
        )

    output_ids = -torch.ones(B, S + max_new_tokens, device=device, dtype=torch.int)
    output_ids[:, :S] = input_ids
    prefill_input_pos = torch.arange(S, device=device, dtype=torch.int)[None, :].expand(
        B, S
    )

    # We need to compute this outside prefill or inductor complains
    prefill_mask = make_prefill_mask(
        seqlens,
        S,
        max_seqlen,
        compile=compile,
    )

    next_token = prefill(
        model=model,
        x=input_ids,
        input_pos=prefill_input_pos,
        prefill_mask=prefill_mask,
        seqlens=seqlens,
        sampling=sampling,
        compile=compile,
    ).clone()
    b_inds = torch.arange(B, device=device)
    output_ids[b_inds, seqlens] = next_token.squeeze(1)

    gen_input_pos = seqlens.clone()
    cur_token = next_token.squeeze(1)

    next_tokens = decode_n_tokens(
        model=model,
        cur_token=cur_token,
        input_pos=gen_input_pos,
        max_new_tokens=max_new_tokens - 1,  # already decoded one
        max_seqlen=max_seqlen,
        sampling=sampling,
        compile=compile,
        stopping_condition=stopping_condition,
    )
    num_new_tokens = next_tokens.shape[1]
    for b in range(B):
        end_idx = seqlens[b] + 1 + num_new_tokens
        output_ids[b, seqlens[b] + 1 : end_idx] = next_tokens[b]

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
    max_seqlen: Optional[int],
    batch_size: int,
    sampling: SamplingConfig,
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

    if max_seqlen is not None:
        assert max_seqlen <= model.config.block_size, (
            f"{max_seqlen} > {model.config.block_size}"
        )
    else:
        max_seqlen = model.config.block_size

    with device:
        model.setup_caches(max_batch_size=batch_size, max_seqlen=max_seqlen)

    stopping_condition = partial(
        contains_word_stopping_condition,
        tokenizer=tokenizer,
        stop_strings=["<|eot_id|>", "<|end_of_text|>"],
    )
    with open(output_file, "a") as f:
        for batch in tqdm(input_iterator, desc="Generating"):
            batch: Batch
            n_trim = batch_size - len(batch.texts)
            if n_trim != 0:
                batch.texts.extend(["pad"] * (n_trim))
                stopping_condition = combine_stopping_conditions(
                    stopping_condition,
                    ignore_batch_inds(
                        device=device,
                        B=batch_size,
                        batch_indices=range(len(batch.texts), batch_size),
                    ),
                )
            device_sync(device=device)
            encoded = tokenize_and_pad(batch.texts, tokenizer, max_seqlen)
            output = generate(
                model=model,
                input_ids=encoded.padded,
                seqlens=encoded.seqlens,
                max_seqlen=max_seqlen,
                max_new_tokens=max_new_tokens,
                device=device,
                compile=compile,
                sampling=sampling,
                stopping_condition=stopping_condition,
            )

            if n_trim != 0:
                output = output[:-n_trim]
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
        default=Path("samples/sample_input.jsonl"),
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=Path,
        default=Path(f"sample_output_{int(time.time())}.jsonl"),
        help="Output jsonl path",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=200, help="Maximum number of new tokens."
    )
    parser.add_argument(
        "--max_seqlen",
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
    max_seqlen: Optional[int] = args.max_seqlen
    batch_size: int = args.batch_size
    top_k: int = args.top_k
    temperature: float = args.temperature
    checkpoint_path: Path = args.checkpoint_path
    device: torch.device = torch.device(args.device)
    compile: bool = args.compile
    sampling = SamplingConfig(top_k=top_k, temperature=temperature)

    args = parser.parse_args()
    main(
        input_file=input_file,
        output_file=output_file,
        max_new_tokens=max_new_tokens,
        max_seqlen=max_seqlen,
        batch_size=batch_size,
        sampling=sampling,
        compile=compile,
        checkpoint_path=checkpoint_path,
        device=device,
    )
