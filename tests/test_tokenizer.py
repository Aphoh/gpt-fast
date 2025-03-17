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
