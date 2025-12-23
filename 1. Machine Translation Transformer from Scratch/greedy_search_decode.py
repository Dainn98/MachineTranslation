import torch
from config import DEVICE, MAX_DECODE_LEN
from mask import make_src_mask, make_tgt_mask

@torch.no_grad()
def greedy_decode(model, src_seq, src_mask, tgt_tok, max_len=MAX_DECODE_LEN):
    model.eval()

    ys = torch.LongTensor([[tgt_tok.word2id[tgt_tok.BOS]]]).to(DEVICE)
    src = src_seq.unsqueeze(0).to(DEVICE)

    for _ in range(max_len):
        tgt_mask = make_tgt_mask(ys)
        out = model(src, ys, make_src_mask(src), tgt_mask)
        next_word = out[:, -1, :].argmax(-1).item()

        ys = torch.cat([ys, torch.tensor([[next_word]]).to(DEVICE)], dim=1)

        if next_word == tgt_tok.word2id[tgt_tok.EOS]:
            break

    return ys[0].cpu().tolist()
