# !pip install sacrebleu
import pandas as pd
import sacrebleu

try:
    cp_path = "/kaggle/working/log/test_predictions.csv"
    df = pd.read_csv(cp_path)

    hyps = df["pred"].astype(str).tolist()  # list[str]
    refs = [[r] for r in df["ground_truth"].astype(str).tolist()]  # list[list[str]]

    bleu = sacrebleu.corpus_bleu(hyps, refs)
    print("Corpus BLEU:", bleu.score)

except FileNotFoundError:
    print("❌ File test_predictions.csv không tồn tại")
except KeyError as e:
    print(f"❌ Thiếu cột trong CSV: {e}")
except Exception as e:
    print("❌ Lỗi khác:", e)
