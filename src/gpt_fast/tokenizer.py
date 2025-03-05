from dataclasses import dataclass
from typing import List
from tokenizers import Tokenizer, Encoding
from pathlib import Path

import torch


class TokenizerInterface:
    def __init__(self, model_path: Path):
        self.model_path = model_path

    def encode(self, text):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def decode(self, tokens):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        raise NotImplementedError("This method should be overridden by subclasses.")

    def decode_batch(self, texts: List[List[int]]) -> List[str]:
        raise NotImplementedError("This method should be overridden by subclasses.")


class HfTokenizerWrapper(TokenizerInterface):
    def __init__(self, tokenizer_path: Path):
        super().__init__(tokenizer_path)
        self.processor: Tokenizer = Tokenizer.from_file(str(tokenizer_path))

    def encode(self, text):
        res = self.processor.encode(text, add_special_tokens=False).ids
        return res

    def decode(self, tokens):
        return self.processor.decode(tokens, skip_special_tokens=False)

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        res: List[Encoding] = self.processor.encode_batch_fast(
            texts, add_special_tokens=False
        )
        return [enc.ids for enc in res]

    def decode_batch(self, ids: List[List[int]]) -> List[str]:
        res = self.processor.decode_batch(ids, skip_special_tokens=False)
        return res


def get_tokenizer(checkpoint_dir: Path):
    """
    Factory function to get the appropriate tokenizer based on the model name.

    Args:
    - tokenizer_model_path (str): The file path to the tokenizer model.
    - model_name (str): The name of the model, used to determine the tokenizer type.

    Returns:
    - TokenizerInterface: An instance of a tokenizer.
    """

    tokenizer_json = checkpoint_dir / "tokenizer.json"
    assert tokenizer_json.is_file(), str(tokenizer_json)
    print("Loading tokenizer from", tokenizer_json)
    return HfTokenizerWrapper(tokenizer_json)


def _round_to_multiple(n: int, multiple: int):
    return ((n + multiple - 1) // multiple) * multiple


@dataclass
class PaddedOutput:
    start_inds: torch.IntTensor
    """[B] tensor of the starting index of each sequence after padding."""
    padded: torch.IntTensor
    """[B, S] tensor of padded sequences."""


def tokenize_and_pad(
    texts: List[str],
    tokenizer: TokenizerInterface,
    max_length: int,
    pad_to_multiple: int = 256,
) -> PaddedOutput:
    """
    Tokenizes a list of texts and pads them to the same length.

    Args:
    - texts (List[str]): A list of texts to tokenize.
    - tokenizer (TokenizerInterface): The tokenizer to use.
    - max_length (int): The maximum length to pad the sequences to.
    - pad_to_multiple (int): The multiple to pad the sequences to.

    Returns:
    - List[List[int]]: A list of tokenized and padded sequences.
    """
    assert max_length % pad_to_multiple == 0, (
        f"max_length={max_length} must be a multiple of pad_to_multiple={pad_to_multiple}"
    )

    tokenized = tokenizer.encode_batch(texts)
    tensors = [torch.tensor(t) for t in tokenized]
    longest_seq = max(len(t) for t in tensors)
    final_len = min(_round_to_multiple(longest_seq, pad_to_multiple), max_length)
    # Pad sequences to the same length
    tensors = [t[:final_len] for t in tensors]
    # keep track of how much padding was added to the left of each sequence
    start_inds = [final_len - t.shape[0] for t in tensors]
    padded = torch.nn.utils.rnn.pad_sequence(
        tensors, batch_first=True, padding_value=0, padding_side="left"
    )
    extra_padding = final_len - padded.shape[1]
    padded = torch.nn.functional.pad(padded, (extra_padding, 0), value=0)
    return PaddedOutput(start_inds=torch.tensor(start_inds), padded=padded)
