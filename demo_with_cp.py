import torch
set_seed()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

(train_en, train_vi), (dev_en, dev_vi), (test_en, test_vi) = load_iwslt15_text()
src_tok = SimpleTokenizer()
tgt_tok = SimpleTokenizer()
src_tok.fit(train_en)
tgt_tok.fit(train_vi)

model_check = Transformer(
        src_tok.vocab_size_(), tgt_tok.vocab_size_(),
        d_model=D_MODEL, N=NUM_LAYERS, heads=NUM_HEADS
    ).to(device)

ckpt_path = "/kaggle/input/logging-mt/iwslt_transformer_v1_best.pt"
state_dict = torch.load(ckpt_path, map_location=device)
model_check.load_state_dict(state_dict)
model_check.eval()

res = evaluate_test_metrics(model_check, test_en, test_vi, src_tok, tgt_tok,max_samples = 10, is_beam = True)
