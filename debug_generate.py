import torch
from gpt_fast.generate import generate, SamplingConfig
from gpt_fast.model import Transformer
from gpt_fast.tokenizer import detokenize_output_ids, get_tokenizer, tokenize_and_pad
from pathlib import Path

from gpt_fast.util import load_model

def test_both_modes(input_text, checkpoint_path):
    # Load model and tokenizer
    model = Transformer.from_name(checkpoint_path.parent.name)
    model = load_model(model, checkpoint_path, precision=torch.bfloat16, device="cuda")
    tokenizer = get_tokenizer(checkpoint_path.parent)
    
    # Setup
    max_seq_length = 512
    with torch.device("cuda"):
        model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)
    encoded = tokenize_and_pad([input_text], tokenizer, max_seq_length, pad_to_multiple=128)
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

    encoded = tokenize_and_pad([input_text], tokenizer, max_seq_length, pad_to_multiple=64)
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


    
    # Compare outputs
    print("NO COMPILE OUTPUT:\n", nocompile_out[0])
    print("WITH COMPILE OUTPUT:\n", compile_out[0])
    print("WITH COMPILE SMALLER PADDING OUTPUT:\n", compile_padding_out[0])

if __name__ == "__main__":
    test_both_modes("Hello, my name is", Path("checkpoints/unsloth/Llama-3.2-1B-Instruct/model.pth"))