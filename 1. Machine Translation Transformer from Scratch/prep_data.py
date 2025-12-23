import torch
from torch.utils.data import Dataset
import os
from config import PATH, MAX_SEQ_LEN

def load_iwslt15_text(path=PATH):
    train_en = open(os.path.join(path, "train.en.txt"), encoding="utf8").read().splitlines()
    train_vi = open(os.path.join(path, "train.vi.txt"), encoding="utf8").read().splitlines()

    dev_en = open(os.path.join(path, "tst2012.en.txt"), encoding="utf8").read().splitlines()
    dev_vi = open(os.path.join(path, "tst2012.vi.txt"), encoding="utf8").read().splitlines()

    test_en = open(os.path.join(path, "tst2013.en.txt"), encoding="utf8").read().splitlines()
    test_vi = open(os.path.join(path, "tst2013.vi.txt"), encoding="utf8").read().splitlines()

    print("Loaded IWSLT15:")
    print(" - Train:", len(train_en))
    print(" - Dev  :", len(dev_en))
    print(" - Test :", len(test_en))

    return (train_en, train_vi), (dev_en, dev_vi), (test_en, test_vi)


class NMTDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, src_tok, tgt_tok, max_len=MAX_SEQ_LEN):
        self.src = src_texts
        self.tgt = tgt_texts
        self.src_tok = src_tok
        self.tgt_tok = tgt_tok
        self.max_len = max_len

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src_ids = self.src_tok.encode(self.src[idx], self.max_len)
        tgt_ids = self.tgt_tok.encode(self.tgt[idx], self.max_len)
        return torch.LongTensor(src_ids), torch.LongTensor(tgt_ids)
