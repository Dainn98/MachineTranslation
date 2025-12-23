from unsloth import FastLanguageModel
import torch
from tqdm import tqdm
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# ------------------------------------------------------------------------
# CẤU HÌNH (Sửa ở đây tùy túi tiền)
# ------------------------------------------------------------------------
max_seq_length = 2048 # Dài quá thì cắt, 2048 là đủ cho đoạn văn y tế rồi
dtype = None          # Để None cho nó tự nhận diện (Float16 cho T4, Bfloat16 cho Ampere)
load_in_4bit = True   # Bắt buộc True để tiết kiệm VRAM

# CHỌN MODEL:
model_name = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"

output_model_name = "output/qwen_mt_3B_v1"

# ------------------------------------------------------------------------
# 1. LOAD MODEL & TOKENIZER
# ------------------------------------------------------------------------
print(f"⏳ Đang tải model {model_name}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 2. GẮN LORA (Cái này giúp model học được mà không tốn nhiều VRAM)
model = FastLanguageModel.get_peft_model(
    model,
    r = 64, # Số càng to càng thông minh nhưng tốn VRAM (16 là chuẩn bài)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 128,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# ------------------------------------------------------------------------
# 3. XỬ LÝ DỮ LIỆU (QUAN TRỌNG)
# ------------------------------------------------------------------------
# Load file data 2 chiều mày vừa tạo
dataset = load_dataset("json", data_files=f"input/final_ultimate_train_utf8.jsonl", split="train")

# Hàm biến đổi format messages thành text thuần để train
def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return {"text": texts}

# Map dữ liệu
print("📝 Đang format dữ liệu...")
dataset = dataset.map(formatting_prompts_func, batched = True)

# ------------------------------------------------------------------------
# 4. CẤU HÌNH TRAIN (Hyperparameters)
# ------------------------------------------------------------------------
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # True nếu muốn train nhanh hơn cho data lớn, data nhỏ thì False cho chắc
    args = TrainingArguments(
        per_device_train_batch_size = 32,    # Tăng lên 4 nếu VRAM còn dư
        gradient_accumulation_steps = 2,    # Tích lũy gradient
        warmup_steps = 100,
        # max_steps = 1500,
        num_train_epochs = 1,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 40,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = f"{output_model_name}",
        report_to = "none",
    ),
)

# ------------------------------------------------------------------------
# 5. BẤM NÚT TRAIN 🚀
# ------------------------------------------------------------------------
print("🚀 Bắt đầu tu luyện...")
trainer_stats = trainer.train()

# ------------------------------------------------------------------------
# 6. SAVE (QUAN TRỌNG NHẤT)
# ------------------------------------------------------------------------

print(f"💾 Đang lưu model vào thư mục: {output_model_name} ...")
model.save_pretrained(output_model_name)
tokenizer.save_pretrained(output_model_name)

# (Tùy chọn) Lưu định dạng GGUF luôn nếu thích (bỏ comment nếu cần)
# model.save_pretrained_gguf(output_model_name, tokenizer, quantization_method = "q4_k_m")

print("✅ XONG PHIM! NHỚ TẢI VỀ NGAY!")