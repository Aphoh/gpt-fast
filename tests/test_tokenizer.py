import torch
from pathlib import Path
from typing import List
from gpt_fast.tokenizer import (
    detokenize_output_ids,
    tokenize_and_pad,
    PaddedOutput,
    TokenizerInterface,
)


class MockTokenizer(TokenizerInterface):
    def __init__(self):
        super().__init__(Path("mock_path"))

    def encode(self, text):
        # Simple encoding: each character becomes its ASCII value
        return [ord(c) for c in text]

    def decode(self, tokens):
        return "".join([chr(t) for t in tokens])

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        return [self.encode(text) for text in texts]

    def decode_batch(self, ids: List[List[int]]) -> List[str]:
        return [self.decode(id_list) for id_list in ids]


def test_tokenize_and_pad_basic():
    tokenizer = MockTokenizer()
    texts = ["hello", "the world"]
    result = tokenize_and_pad(texts, tokenizer, max_length=256, pad_to_multiple=8)

    assert isinstance(result, PaddedOutput)
    assert result.padded.shape[0] == len(texts)  # Batch size
    assert result.padded.shape[1] == 16  # Padded to multiple of 8
    assert result.seqlens.tolist() == [len(t) for t in texts]

    # Check padding on the right
    for i, t in enumerate(texts):
        assert torch.all(result.padded[i, len(t) :] == -1)

    # Check actual content
    assert detokenize_output_ids(result.padded, tokenizer) == texts


def test_tokenize_and_pad_near_maxseqlen():
    tokenizer = MockTokenizer()
    max_length = 128
    texts = ["hello", "c" + ("a" * 127)]
    result = tokenize_and_pad(
        texts, tokenizer, max_length=max_length, pad_to_multiple=8, min_new_tokens=1
    )

    assert isinstance(result, PaddedOutput)
    assert result.padded.shape[0] == len(texts)  # Batch size
    assert result.padded.shape[1] == 128  # Padded to multiple of 8
    assert result.seqlens.tolist() == [len("hello"), 127]

    assert torch.all(result.padded[0, 5:] == -1)  # Padding on the right
    assert torch.all(result.padded[1, :-1] != -1)  # All but the last aren't padding
    assert torch.all(result.padded[:, -1] == -1)  # The last is padding

    # Check actual content
    detok = detokenize_output_ids(result.padded, tokenizer)
    assert detok[0] == "hello"
    assert detok[1] == ("a" * 127)
