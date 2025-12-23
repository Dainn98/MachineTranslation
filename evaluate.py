import os
import torch
import torch.nn.functional as F
import sacrebleu
import csv

from config import DEVICE, BATCH_SIZE
from mask import make_src_mask, make_tgt_mask
from greedy_search_decode import greedy_decode
from beam_search_decode import beam_decode
from prep_data import NMTDataset
from collate import collate_batch

def evaluate_test_metrics(model, test_src, test_tgt,
                        src_tok, tgt_tok, max_samples=None,
                        bpe_type="sentencepiece",
                        save_dir: str = "./log",
                        log_name: str = "test_predictions.csv",
                        is_beam = False):
    model.eval()
    # ====== Prepare log ======
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, log_name)

    log_rows = []
    log_rows.append(["input", "ground_truth", "pred", "bleu_score"])
    
    # ====== BLEU ======
    hyps = []
    refs = []

    if max_samples is None:
        max_samples = len(test_src)
    with torch.no_grad():
        for i in range(max_samples):
            # ====== SOURCE ======
            src_text = test_src[i]
            tgt_text = test_tgt[i]
    
            # encode EN
            src_ids = torch.LongTensor(src_tok.encode(src_text)).unsqueeze(0).to(DEVICE)
            src_mask = make_src_mask(src_ids)
    
            # ====== GREEDY DECODE ======
            if is_beam == False:
                out_ids = greedy_decode(model, src_ids[0], src_mask[0], tgt_tok)
            else:
                out_ids = beam_decode(model, src_ids[0], src_mask[0], tgt_tok)
            
            hyp = tgt_tok.decode(out_ids)
    
            # ====== DETOKENIZE ======
            if bpe_type == "sentencepiece":
                hyp = hyp.replace("▁", " ").strip()
                ref = tgt_text.replace("▁", " ").strip()
            else:
                ref = tgt_text.strip()
    
            hyps.append(hyp)
            refs.append([ref])
    
            # ====== Sentence BLEU ======
            sent_bleu = sacrebleu.sentence_bleu(hyp, [ref]).score
    
            # ====== Log row ======
            log_rows.append([
                src_text,
                ref,
                hyp,
                round(sent_bleu, 4)
            ])
    
    # ====== Write CSV ======
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(log_rows)
        
    bleu = sacrebleu.corpus_bleu(hyps, refs).score

    # ================= ACC + PPL =================
    pad_id = tgt_tok.pad_id()
    total_correct = 0
    total_tokens = 0
    total_loss = 0.0

    test_ds = NMTDataset(test_src, test_tgt, src_tok, tgt_tok)
    loader = torch.utils.data.DataLoader(test_ds, batch_size = BATCH_SIZE, shuffle = False, collate_fn = collate_batch)
    
    for src, tgt in loader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_in   = tgt[:, :-1]
        tgt_gold = tgt[:, 1:]

        src_mask = make_src_mask(src)
        tgt_mask = make_tgt_mask(tgt_in)

        logits = model(src, tgt_in, src_mask, tgt_mask)
        # [B, T, V]

        vocab_size = logits.size(-1)
        logits = logits.reshape(-1, vocab_size)
        tgt_gold = tgt_gold.reshape(-1)

        loss = F.cross_entropy(
            logits,
            tgt_gold,
            ignore_index=pad_id,
            reduction="sum"
        )
        total_loss += loss.item()

        preds = logits.argmax(dim=-1)
        mask = tgt_gold != pad_id

        total_correct += (preds[mask] == tgt_gold[mask]).sum().item()
        total_tokens  += mask.sum().item()

    avg_loss = total_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    acc = total_correct / total_tokens

    #Console
    print(
        f"TEST BLEU: {bleu:.4f} | "
        f"TEST PPL: {ppl:.4f} | "
        f"TEST ACC: {acc:.4f}"
    )
    print(f"Prediction log saved to: {log_path}")

    return {
        "bleu": bleu,
        "ppl": ppl,
        "acc": acc
    }
