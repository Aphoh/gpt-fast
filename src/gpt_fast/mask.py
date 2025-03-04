import torch

from torch.nn.attention.flex_attention import create_block_mask


create_block_mask_complied = torch.compile(create_block_mask)
