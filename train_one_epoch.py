import torch
from config import DEVICE
from mask import make_src_mask, make_tgt_mask

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for src, tgt in loader:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)

        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        src_mask = make_src_mask(src).to(DEVICE)
        tgt_mask = make_tgt_mask(tgt_in).to(DEVICE)

        pred = model(src, tgt_in, src_mask, tgt_mask)
        pred = pred.reshape(-1, pred.size(-1))
        tgt_out = tgt_out.reshape(-1)

        loss = criterion(pred, tgt_out)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)
