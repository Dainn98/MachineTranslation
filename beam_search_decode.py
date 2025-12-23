import torch
import torch.nn.functional as F
from config import DEVICE, BEAM_SIZE, MAX_DECODE_LEN, LENGTH_PENALTY
from mask import make_src_mask, make_tgt_mask

@torch.no_grad()
def beam_decode(
    model,
    src_seq,
    src_mask,
    tgt_tok,
    beam_size=BEAM_SIZE,
    max_len=MAX_DECODE_LEN,
    alpha=LENGTH_PENALTY
):
    model.eval()

    BOS = tgt_tok.word2id[tgt_tok.BOS]
    EOS = tgt_tok.word2id[tgt_tok.EOS]

    # src: [1, S]
    src = src_seq.unsqueeze(0).to(DEVICE)
    src_mask = src_mask.unsqueeze(0).to(DEVICE)  # [1,1,1,S]

    # beam = (log_prob, token_ids)
    beams = [(0.0, [BOS])]
    completed = []

    for _ in range(max_len):
        new_beams = []

        for log_p, seq in beams:
            if seq[-1] == EOS:
                completed.append((log_p, seq))
                continue

            tgt = torch.LongTensor(seq).unsqueeze(0).to(DEVICE)
            tgt_mask = make_tgt_mask(tgt)

            logits = model(src, tgt, src_mask, tgt_mask)
            log_probs = F.log_softmax(logits[:, -1, :], dim=-1).squeeze(0)

            topk_log_p, topk_ids = torch.topk(log_probs, beam_size)

            for k in range(beam_size):
                new_seq = seq + [topk_ids[k].item()]
                new_log_p = log_p + topk_log_p[k].item()
                new_beams.append((new_log_p, new_seq))

        # giữ top beam_size
        beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_size]

        if len(completed) >= beam_size:
            break

    candidates = completed if completed else beams

    def lp(length):
        return ((5 + length) / 6) ** alpha

    best = max(
        candidates,
        key=lambda x: x[0] / lp(len(x[1]))
    )

    return best[1]
