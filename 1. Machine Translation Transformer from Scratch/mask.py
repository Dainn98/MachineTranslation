import torch
from config import DEVICE
def make_src_mask(src):
    return (src != 0).unsqueeze(1).unsqueeze(2) # [B,1,1,S]

def make_tgt_mask(tgt):
    T = tgt.size(1)
    pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2) # [B,1,1,T]
    seq_mask = torch.tril(torch.ones((T, T), device=DEVICE)).bool()
    return pad_mask & seq_mask # broadcast → [B,1,T,T]
