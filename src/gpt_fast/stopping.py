from functools import partial
from typing import List, Sequence
from gpt_fast.tokenizer import TokenizerInterface
import torch
from typing import Protocol, runtime_checkable


@runtime_checkable
class StoppingCondition(Protocol):
    """Protocol for stopping conditions in text generation.

    Stopping conditions determine when to stop generating tokens for each sequence
    in a batch.
    """

    def __call__(
        self,
        output_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Check if sequences should stop generating.

        Args:
            output_tokens: Tensor of shape (B, seqlen) containing generated tokens

        Returns:
            stop_map: Boolean tensor of shape (B,) where True indicates stopping
        """
        ...


def contains_word_stopping_condition(
    output_tokens: torch.Tensor,
    *,
    tokenizer: TokenizerInterface,
    stop_strings: List[List[str]],
    token_window: int = 10,
):
    B, _ = output_tokens.shape
    check_tokens = []
    for b in range(B):
        check_tokens.append(output_tokens[b, -token_window:].clamp(min=0).tolist())
    stop_map = torch.zeros(B, dtype=torch.bool, device=output_tokens.device)
    check_strings = tokenizer.decode_batch(check_tokens)
    for b in range(B):
        if b < len(stop_strings):
            for stop_string in stop_strings[b]:
                if stop_string in check_strings[b]:
                    stop_map[b] = True
                    break
    return stop_map


def constant_stop_map(
    _output_tokens: torch.Tensor, *, stop_map: torch.Tensor
) -> torch.Tensor:
    return stop_map


def ignore_batch_inds(
    device: torch.device, B: int, batch_indices: Sequence[int]
) -> StoppingCondition:
    stop_map = torch.zeros(B, dtype=torch.bool, device=device)
    for b in batch_indices:
        stop_map[b] = True
    return partial(constant_stop_map, stop_map=stop_map)


def combine_stopping_conditions(
    *stop_conditions: StoppingCondition,
) -> StoppingCondition:
    def combined_stop_condition(output_tokens: torch.Tensor) -> torch.Tensor:
        stop_map = torch.zeros(
            output_tokens.shape[0], dtype=torch.bool, device=output_tokens.device
        )
        for stop_condition in stop_conditions:
            stop_map |= stop_condition(output_tokens)
        return stop_map

    return combined_stop_condition
