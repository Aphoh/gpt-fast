import pytest
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
    texts = ["hello"]
    result = tokenize_and_pad(texts, tokenizer, max_length=256, pad_to_multiple=8)

    assert isinstance(result, PaddedOutput)
    assert result.padded.shape[0] == 1  # Batch size
    assert result.padded.shape[1] == 8  # Padded to multiple of 8
    assert result.start_inds.shape[0] == 1
    assert result.start_inds[0].item() == 3  # 8 - 5 (length of "hello")

    # Check padding on the left
    assert torch.all(result.padded[0, :3] == 0)

    # Check actual content
    expected_tokens = torch.tensor([ord(c) for c in "hello"])
    assert torch.all(result.padded[0, 3:] == expected_tokens)


def test_tokenize_and_pad_no_multiple():
    tokenizer = MockTokenizer()
    texts = ["hello", "world", "longer example"]  # lengths: 5, 5, 14
    result = tokenize_and_pad(texts, tokenizer, max_length=256, pad_to_multiple=None)

    # Check that padded shape is exactly the length of the longest sequence (14)
    assert result.padded.shape[0] == 3  # Batch size
    assert result.padded.shape[1] == 14  # Length of "longer example" with no additional padding

    # Check start indices
    assert result.start_inds[0].item() == 9  # 14 - 5 (length of "hello")
    assert result.start_inds[1].item() == 9  # 14 - 5 (length of "world")
    assert result.start_inds[2].item() == 0  # 14 - 14 (length of "longer example")

    # Verify the content is correctly positioned with padding on the left
    for i, text in enumerate(texts):
        expected_tokens = torch.tensor([ord(c) for c in text])
        actual_tokens = result.padded[i, result.start_inds[i]:]
        assert torch.all(actual_tokens == expected_tokens)
        
        # Check that padding (zeros) is only on the left
        if result.start_inds[i] > 0:
            assert torch.all(result.padded[i, :result.start_inds[i]] == 0)


def test_tokenize_and_pad_multiple_texts():
    tokenizer = MockTokenizer()
    texts = ["hello", "world", "longer example"]
    result = tokenize_and_pad(texts, tokenizer, max_length=256, pad_to_multiple=8)

    assert result.padded.shape[0] == 3  # Batch size
    assert (
        result.padded.shape[1] == 16
    )  # Padded to nearest multiple of 8 (length of "longer example")

    # Check start indices
    assert result.start_inds[0].item() == 11  # 16 - 5 (length of "hello")
    assert result.start_inds[1].item() == 11  # 16 - 5 (length of "world")
    assert result.start_inds[2].item() == 2  # 16 - 14 (length of "longer example")


def test_tokenize_and_pad_empty_text():
    tokenizer = MockTokenizer()
    texts = ["", "hello"]
    result = tokenize_and_pad(texts, tokenizer, max_length=256, pad_to_multiple=8)

    assert result.padded.shape[0] == 2
    assert result.padded.shape[1] == 8  # Padded to multiple of 8

    # Check start indices
    assert result.start_inds[0].item() == 8  # 8 - 0 (empty string)
    assert result.start_inds[1].item() == 3  # 8 - 5 (length of "hello")


def test_tokenize_and_pad_max_length():
    tokenizer = MockTokenizer()
    long_text = "a" * 20  # Text longer than max_length
    texts = [long_text]
    max_length = 16

    result = tokenize_and_pad(
        texts, tokenizer, max_length=max_length, pad_to_multiple=8
    )

    assert result.padded.shape[1] == max_length
    # Text should be truncated to max_length
    assert (
        result.start_inds[0].item() == 0
    )  # No left padding since text was truncated to max_length


def test_tokenize_and_pad_different_multiples():
    tokenizer = MockTokenizer()
    texts = ["testing"]

    # Test with pad_to_multiple=4
    result = tokenize_and_pad(texts, tokenizer, max_length=256, pad_to_multiple=4)
    assert result.padded.shape[1] == 8  # Next multiple of 4 after 7

    # Test with pad_to_multiple=16
    result = tokenize_and_pad(texts, tokenizer, max_length=256, pad_to_multiple=16)
    assert result.padded.shape[1] == 16  # Next multiple of 16 after 7


def test_tokenize_and_pad_invalid_max_length():
    tokenizer = MockTokenizer()
    texts = ["test"]

    with pytest.raises(AssertionError):
        # max_length is not a multiple of pad_to_multiple
        tokenize_and_pad(texts, tokenizer, max_length=10, pad_to_multiple=8)


def test_detokenize_output_ids():
    tokenizer = MockTokenizer()
    texts = ["".join(["test"] * (a + 2)) for a in range(4)]
    encoded = tokenize_and_pad(texts, tokenizer, 256, pad_to_multiple=32)
    assert list(encoded.padded.shape) == [4, 32]
    encoded.padded[0, -4:] = -1  # should leave "test"
    encoded.padded[1, -8:] = -1  # should leave "test"
    print(encoded.start_inds)
    detok = detokenize_output_ids(encoded.start_inds, encoded.padded, tokenizer)
    assert detok == [
        "test",
        "test",
        "test" * 4,
        "test" * 5,
    ]
