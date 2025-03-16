import torch
from gpt_fast.generate import generate, SamplingConfig
from gpt_fast.model import Transformer
from gpt_fast.tokenizer import detokenize_output_ids, get_tokenizer, tokenize_and_pad
from pathlib import Path

from gpt_fast.util import load_model

def test_both_modes(input_texts, checkpoint_path):
    # Load model and tokenizer
    model = Transformer.from_name(checkpoint_path.parent.name)
    model = load_model(model, checkpoint_path, precision=torch.bfloat16, device="cuda")
    tokenizer = get_tokenizer(checkpoint_path.parent)
    
    # Setup
    max_seq_length = 512
    with torch.device("cuda"):
        model.setup_caches(max_batch_size=len(input_texts), max_seq_length=max_seq_length)
    encoded = tokenize_and_pad(input_texts, tokenizer, max_seq_length, pad_to_multiple=128)
    sampling = SamplingConfig(top_k=None, temperature=0.0)
    
    # Test with compile=False
    print("TESTING WITHOUT COMPILE")
    output = generate(
        model=model, 
        input_ids=encoded.padded,
        start_inds=encoded.start_inds,
        max_seq_length=max_seq_length,
        max_new_tokens=30,
        device="cuda",
        compile=False,
        sampling=sampling
    )
    nocompile_out = detokenize_output_ids(encoded.start_inds, output, tokenizer)
    
    # Test with compile=True
    print("TESTING WITH COMPILE")
    output = generate(
        model=model, 
        input_ids=encoded.padded,
        start_inds=encoded.start_inds,
        max_seq_length=max_seq_length,
        max_new_tokens=30,
        device="cuda",
        compile=True,
        sampling=sampling
    )
    compile_out = detokenize_output_ids(encoded.start_inds, output, tokenizer)

    encoded = tokenize_and_pad(input_texts, tokenizer, max_seq_length, pad_to_multiple=64)
    # Test with compile=True, less padding
    print("TESTING WITH COMPILE, PADDING TO 64")
    output = generate(
        model=model, 
        input_ids=encoded.padded,
        start_inds=encoded.start_inds,
        max_seq_length=max_seq_length,
        max_new_tokens=30,
        device="cuda",
        compile=True,
        sampling=sampling
    )
    compile_padding_out = detokenize_output_ids(encoded.start_inds, output, tokenizer)

    encoded = tokenize_and_pad(input_texts, tokenizer, max_seq_length, pad_to_multiple=None)
    # Test with compile=True, less padding
    print("TESTING WITH COMPILE, NO PADDING")
    output = generate(
        model=model, 
        input_ids=encoded.padded,
        start_inds=encoded.start_inds,
        max_seq_length=max_seq_length,
        max_new_tokens=30,
        device="cuda",
        compile=True,
        sampling=sampling
    )
    print(output)
    compile_padding_1_out = detokenize_output_ids(encoded.start_inds, output, tokenizer)

    
    # Compare outputs
    print("NO COMPILE OUTPUT:\n", nocompile_out)
    print("WITH COMPILE OUTPUT:\n", compile_out)
    print("WITH COMPILE 64 PADDING OUTPUT:\n", compile_padding_out)
    print("WITH COMPILE 1 PADDING OUTPUT:\n", compile_padding_1_out)

if __name__ == "__main__":
    texts = ["short input", "This is a longer input that should be able to generate more text. " * 5]
    test_both_modes(texts, Path("checkpoints/unsloth/Llama-3.2-1B-Instruct/model.pth"))