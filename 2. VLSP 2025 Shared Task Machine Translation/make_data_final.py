import json
import random

# -----------------------------------------------------------
# CẤU HÌNH INPUT
# -----------------------------------------------------------
vi_file_path = "input/clean_train.vi.txt"       # File tiếng Việt
en_file_path = "input/clean_train.en.txt"       # File tiếng Anh
jsonl_gloss_path = "input/bidirectional_train_data.jsonl" # File từ điển/viết tắt
output_file = "input/final_ultimate_train.jsonl" # Tên file kết quả

# --- CẤU HÌNH SỐ LƯỢNG ---
SAMPLE_SIZE = 100000 

# Hệ số nhân bản cho file từ điển/viết tắt
GLOSS_MULTIPLIER = 20

training_data = []

# --- PHẦN 1: ĐỌC 2 FILE TEXT SONG SONG ---
print("⏳ Đang đọc 2 file text...")
with open(vi_file_path, "r", encoding="utf-8") as f_vi, \
     open(en_file_path, "r", encoding="utf-8") as f_en:
    
    vi_lines = [line.strip() for line in f_vi]
    en_lines = [line.strip() for line in f_en]

# Kiểm tra lệch dòng
if len(vi_lines) != len(en_lines):
    print(f"⚠️ CẢNH BÁO: Số dòng không khớp! VI: {len(vi_lines)} - EN: {len(en_lines)}")
    # Lấy số dòng nhỏ nhất để zip
    min_len = min(len(vi_lines), len(en_lines))
    vi_lines = vi_lines[:min_len]
    en_lines = en_lines[:min_len]
else:
    print(f"✅ Số dòng khớp nhau tuyệt đối: {len(vi_lines)} dòng.")

# Ghép cặp
paired_lines = list(zip(en_lines, vi_lines))

# Lấy mẫu ngẫu nhiên (Sampling)
if SAMPLE_SIZE and SAMPLE_SIZE < len(paired_lines):
    print(f"✂️ Lấy ngẫu nhiên {SAMPLE_SIZE} cặp câu để train...")
    sampled_pairs = random.sample(paired_lines, SAMPLE_SIZE)
else:
    sampled_pairs = paired_lines

# Convert sang format Chat
print("🔄 Đang convert sang format Qwen...")
for en_text, vi_text in sampled_pairs:
    if not en_text or not vi_text: continue # Bỏ qua dòng trống
    
    # Chiều En -> Vi
    training_data.append({
        "messages": [
            {"role": "system", "content": "You are a professional medical translator."},
            {"role": "user", "content": f"Translate to Vietnamese: {en_text}"},
            {"role": "assistant", "content": vi_text}
        ]
    })
    
    # Chiều Vi -> En (Dạy luôn chiều ngược)
    training_data.append({
        "messages": [
            {"role": "system", "content": "You are a professional medical translator."},
            {"role": "user", "content": f"Translate to English: {vi_text}"},
            {"role": "assistant", "content": en_text}
        ]
    })

print(f"📊 Đã xong phần data sạch. Số mẫu: {len(training_data)}")

# --- PHẦN 2: TRỘN VÀ NHÂN BẢN FILE TỪ ĐIỂN & VIẾT TẮT ---
print("⏳ Đang trộn file từ điển & viết tắt...")
gloss_data = []
try:
    with open(jsonl_gloss_path, "r", encoding="utf-8") as f:
        for line in f:
            gloss_data.append(json.loads(line))
            
    print(f"💪 Nhân bản data GLOSS lên {GLOSS_MULTIPLIER} lần ...")
    for _ in range(GLOSS_MULTIPLIER):
        training_data.extend(gloss_data)
        
except FileNotFoundError:
    print("❌ LỖI: Không tìm thấy file glossory.")

# --- PHẦN 3: LƯU FILE CUỐI CÙNG ---
print("🔀 Đang trộn đều (Shuffle) lần cuối...")
random.shuffle(training_data)

with open(output_file, "w", encoding="utf-8") as f:
    for entry in training_data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"🎉 XONG! File train final: '{output_file}'")
print(f"📈 Tổng số lượng mẫu training: {len(training_data)}")