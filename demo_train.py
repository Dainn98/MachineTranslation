import torch

from config import *
from helper import set_seed
from prep_data import load_iwslt15_text
from train_full_pipeline import train_pipeline
from evaluate import evaluate_test_metrics

set_seed()

print(f"Using device: {DEVICE}")

print(f'='*80)
(train_en, train_vi), (dev_en, dev_vi), (test_en, test_vi) = load_iwslt15_text(PATH)

N_TRAIN = 100
N_DEV   = 20

train_en_small = train_en[:N_TRAIN]
train_vi_small = train_vi[:N_TRAIN]

dev_en_small = dev_en[:N_DEV]
dev_vi_small = dev_vi[:N_DEV]

print(f'='*80)
# model_iwslt, tok_iwslt_en, tok_iwslt_vi = train_pipeline(train_en, train_vi, dev_en, dev_vi, model_name=MODEL_NAME)
model_iwslt, tok_iwslt_en, tok_iwslt_vi = train_pipeline(
    train_en_small,
    train_vi_small,
    dev_en_small,
    dev_vi_small,
    model_name=MODEL_NAME
)

print(f'='*80)
res = evaluate_test_metrics(model_iwslt, test_en, test_vi, tok_iwslt_en, tok_iwslt_vi,max_samples = 10,is_beam = False) 
res = evaluate_test_metrics(model_iwslt, test_en, test_vi, tok_iwslt_en, tok_iwslt_vi, max_samples = 2,is_beam = True) 
