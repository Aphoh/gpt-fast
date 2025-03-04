import json
import os
import sentencepiece as spm
import tiktoken
from tiktoken.load import load_tiktoken_bpe
from tokenizers import Tokenizer
from pathlib import Path
from typing import Dict


class TokenizerInterface:
    def __init__(self, model_path: Path):
        self.model_path = model_path

    def encode(self, text):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def decode(self, tokens):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def bos_id(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def eos_id(self):
        raise NotImplementedError("This method should be overridden by subclasses.")


class SentencePieceWrapper(TokenizerInterface):
    def __init__(self, model_path):
        super().__init__(model_path)
        self.processor = spm.SentencePieceProcessor(str(model_path))

    def encode(self, text):
        return self.processor.EncodeAsIds(text)

    def decode(self, tokens):
        return self.processor.DecodeIds(tokens)

    def bos_id(self):
        return self.processor.bos_id()

    def eos_id(self):
        return self.processor.eos_id()


class HfTokenizerWrapper(TokenizerInterface):
    def __init__(self, checkpoint_dir: Path):
        super().__init__(checkpoint_dir / "tokenizer.json")
        tokenizer_path = checkpoint_dir / "tokenizer.json"
        assert tokenizer_path.is_file(), str(tokenizer_path)
        self.processor: Tokenizer = Tokenizer.from_file(str(tokenizer_path))

        config_loc: Path = checkpoint_dir / "tokenizer_config.json"
        config_json = json.loads(config_loc.read_text())
        self.bos_token_str = config_json["bos_token"]
        self.eos_token_str = config_json["eos_token"]
        self.bos_token_id = self.processor.token_to_id(self.bos_token_str)
        self.eos_token_id = self.processor.token_to_id(self.eos_token_str)

    def encode(self, text):
        res = self.processor.encode(text).ids
        return res

    def decode(self, tokens):
        return self.processor.decode(tokens)

    def bos_id(self):
        return self.bos_token_id

    def eos_id(self):
        return self.eos_token_id


class TiktokenWrapper(TokenizerInterface):
    """
    Tokenizing and encoding/decoding text using the Tiktoken tokenizer.
    """

    special_tokens: Dict[str, int]

    num_reserved_special_tokens = 256

    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

    def __init__(self, model_path):
        super().__init__(model_path)
        assert os.path.isfile(model_path), str(model_path)
        mergeable_ranks = load_tiktoken_bpe(str(model_path))
        num_base_tokens = len(mergeable_ranks)
        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",  # end of turn
        ] + [
            f"<|reserved_special_token_{i}|>"
            for i in range(5, self.num_reserved_special_tokens - 5)
        ]
        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        # BOS / EOS token IDs
        self._bos_id: int = self.special_tokens["<|begin_of_text|>"]
        self._eos_id: int = self.special_tokens["<|end_of_text|>"]

    def encode(self, text):
        return self.model.encode(text)

    def decode(self, tokens):
        return self.model.decode(tokens)

    def bos_id(self):
        return self._bos_id

    def eos_id(self):
        return self._eos_id


def get_tokenizer(checkpoint_dir: Path):
    """
    Factory function to get the appropriate tokenizer based on the model name.

    Args:
    - tokenizer_model_path (str): The file path to the tokenizer model.
    - model_name (str): The name of the model, used to determine the tokenizer type.

    Returns:
    - TokenizerInterface: An instance of a tokenizer.
    """

    tokenizer_model_path = checkpoint_dir / "tokenizer.model"
    tokenizer_config_path = checkpoint_dir / "tokenizer_config.json"
    if tokenizer_model_path.is_file():
        print("Found tokenizer.model")
        if "llama-3" in str(checkpoint_dir).lower():
            return TiktokenWrapper(tokenizer_model_path)
        else:
            return SentencePieceWrapper(tokenizer_model_path)
    elif tokenizer_config_path.is_file():
        print("Found tokenizer_config.json")
        return HfTokenizerWrapper(checkpoint_dir)
    else:
        raise ValueError(f"Tokenizer not found in {checkpoint_dir}")
