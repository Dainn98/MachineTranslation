import torch.nn as nn
def collate_batch(batch):
    src, tgt = zip(*batch)
    src = nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=0)
    tgt = nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=0)
    return src, tgt
