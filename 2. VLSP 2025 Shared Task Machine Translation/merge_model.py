from unsloth import FastLanguageModel
import torch

# Load model Unsloth cũ của mày
model_path = "output/qwen_mt_3B_v1" 
print("⏳ Đang load để Merge...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# Merge ra thư mục mới (dạng 16bit chuẩn)
print("💾 Đang lưu model chuẩn (Merged) vào 'output/merged_qwen_3b'...")
model.save_pretrained_merged("output/merged_qwen_3b", tokenizer, save_method = "merged_16bit")
print("✅ Xong! Giờ dùng cái 'output/merged_qwen_3b' này để chạy inference bao mượt!")