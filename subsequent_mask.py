import numpy as np
import torch

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')#1채운 배열 마지막 삼각형을 0으로 채운다 
    return torch.from_numpy(subsequent_mask) == 0
