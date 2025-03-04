# gpt-fast
Pytorch-only library for efficient single batch and multi-batch inference, useable as a standalone library.

Featuring:
1. Very low latency
2. <1000 lines of python
3. No huggingface transformers
4. Tensor parallelism
5. Supports Nvidia and AMD GPUs

## Supported Models

* Most Llama models
* Starcoder2-3b

## Installation/usage
First install astral's UV

Then you can run generation with `uv run -m gpt_fast.generate --checkpoint_path ...`

## Downloading Weights
Models tested/supported
```text
tinyllamas/stories{15,42,100}
openlm-research/open_llama_7b
meta-llama/Llama-2-7b-chat-hf
meta-llama/Llama-2-13b-chat-hf
meta-llama/Llama-2-70b-chat-hf
codellama/CodeLlama-7b-Python-hf
codellama/CodeLlama-34b-Python-hf
mistralai/Mistral-7B-v0.1
mistralai/Mistral-7B-Instruct-v0.1
mistralai/Mistral-7B-Instruct-v0.2
meta-llama/Meta-Llama-3-8B
meta-llama/Meta-Llama-3.1-8B
meta-llama/Meta-Llama-3.1-70B
meta-llama/Meta-Llama-3.1-405B
```

To convert Llama-2-7b-chat-hf
```bash
export MODEL_REPO=meta-llama/Llama-2-7b-chat-hf
./scripts/prepare.sh $MODEL_REPO
```

## Tensor Parallelism
```bash
ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=2 generate.py --compile --checkpoint_path checkpoints/$MODEL_REPO/model.pth
```
