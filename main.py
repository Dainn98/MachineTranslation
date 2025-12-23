
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

print(f'='*80)
model_iwslt, tok_iwslt_en, tok_iwslt_vi = train_pipeline(train_en, train_vi, dev_en, dev_vi, model_name=MODEL_NAME)


if IS_BEAM:
    print(f'Beam decode')
    res = evaluate_test_metrics(model_iwslt, test_en, test_vi, tok_iwslt_en, tok_iwslt_vi,is_beam = True)     
else:
    print(f'Greedy decode')
    res = evaluate_test_metrics(model_iwslt, test_en, test_vi, tok_iwslt_en, tok_iwslt_vi,is_beam = False) 
        
