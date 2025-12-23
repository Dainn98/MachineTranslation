import torch
import torch.nn as nn
from config import DEVICE, MODEL_NAME, EPOCHS, BATCH_SIZE, PATIENCE, D_MODEL, NUM_LAYERS, NUM_HEADS
from transformer import Transformer
from tokenizer import SimpleTokenizer
from prep_data import NMTDataset
from collate import collate_batch
from train_one_epoch import train_one_epoch
from mask import make_src_mask, make_tgt_mask
def pretty_params(n):
    return f"{n/1e6:.2f}M"

def train_pipeline(train_src, train_tgt, val_src, val_tgt,
                   model_name=MODEL_NAME, epochs=EPOCHS, batch_size=BATCH_SIZE,
                   patience=PATIENCE):

    # === tokenizer ===
    src_tok = SimpleTokenizer()
    tgt_tok = SimpleTokenizer()
    src_tok.fit(train_src)
    tgt_tok.fit(train_tgt)

    # === datasets ===
    train_ds = NMTDataset(train_src, train_tgt, src_tok, tgt_tok)
    val_ds   = NMTDataset(val_src,   val_tgt,   src_tok, tgt_tok)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
    )

    # === model ===
    model = Transformer(
        src_tok.vocab_size_(), tgt_tok.vocab_size_(),
        d_model=D_MODEL, N=NUM_LAYERS, heads=NUM_HEADS
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'trainable_params:',pretty_params(trainable_params))
    print(f'total_params:',pretty_params(total_params))

    print(f'vocab',src_tok.vocab_size_(), tgt_tok.vocab_size_())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=src_tok.word2id[src_tok.PAD])

    # === Early Stopping state (dựa trên loss) ===
    best_val_loss = float("inf")
    patience_counter = 0
    best_path = f"{model_name}_best.pt"

    # === training loop ===
    for ep in range(epochs):

        # ========== TRAIN ==========
        model.train()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)

        # ========== VALIDATION LOSS ==========
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(DEVICE), tgt.to(DEVICE)
                src_mask = make_src_mask(src)
                tgt_input = tgt[:, :-1]     # input
                tgt_output = tgt[:, 1:]     # shift for loss
                tgt_mask = make_tgt_mask(tgt_input)

                logits = model(src, tgt_input, src_mask, tgt_mask)

                vocab_size = logits.shape[-1]
                loss = criterion(
                    logits.reshape(-1, vocab_size),
                    tgt_output.reshape(-1)
                )
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"\nEpoch {ep+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # ===== EARLY STOPPING BASED ON LOSS =====
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
            print(f"✔️  Validation loss improved — model saved!")
        else:
            patience_counter += 1
            print(f"⚠️  Loss did not improve. Patience = {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("⛔ Early stopping triggered (no loss improvement).")
                break

    print("\nTraining completed.")
    print(f"🥇 Best Val Loss: {best_val_loss:.4f}")
    print(f"Model saved at: {best_path}")

    # load best model before returning
    model.load_state_dict(torch.load(best_path))

    return model, src_tok, tgt_tok
