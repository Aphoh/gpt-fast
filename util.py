import torch

from model import Transformer

def load_model(model: Transformer, checkpoint_path: str, precision: torch.dtype, device: torch.device) -> Transformer:
    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    if model.config.tie_embedding_weights:
        checkpoint["output.weight"] = checkpoint["tok_embeddings.weight"]
    model.load_state_dict(checkpoint, assign=True)
    model = model.to(dtype=precision, device=device)
    return model