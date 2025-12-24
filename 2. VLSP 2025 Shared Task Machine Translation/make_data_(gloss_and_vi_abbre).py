import json
import random

# Load data
with open("input/simple_medical_glossary.json", "r", encoding="utf-8") as f:
    en_vi_dict = json.load(f)

with open("input/vi_abbre.json", "r", encoding="utf-8") as f:
    abbr_dict = json.load(f)

training_data = []

# --- PHẦN 1: Dạy từ vựng ---
for en, vi in en_vi_dict.items():
    # En -> Vi
    training_data.append({
        "messages": [
            {"role": "system", "content": "You are a professional medical translator."},
            {"role": "user", "content": f"Translate to Vietnamese: {en}"},
            {"role": "assistant", "content": vi}
        ]
    })
    
    if len(en) <= 4:
        training_data.append({
            "messages": [
                {"role": "system", "content": "You are a professional medical translator."},
                {"role": "user", "content": f"Translate to Vietnamese: {en.upper()}"},
                {"role": "assistant", "content": vi}
            ]
        })
    
    # Vi -> En
    training_data.append({
        "messages": [
            {"role": "system", "content": "You are a professional medical translator."},
            {"role": "user", "content": f"Translate to English: {vi}"},
            {"role": "assistant", "content": en}
        ]
    })

# --- PHẦN 2: Dạy viết tắt (QUAN TRỌNG: NHÂN BẢN X10 LẦN) ---
for abbr, full in abbr_dict.items():
    for i in range(12):
        if i < 6: abbr = abbr.upper()
        else: abbr = abbr.lower()
        training_data.append({
            "messages": [
                {"role": "system", "content": "Explain this Vietnamese medical abbreviation."},
                {"role": "user", "content": f"Nghĩa của từ viết tắt '{abbr}' là gì?"},
                {"role": "assistant", "content": full}
            ]
        })
        
        # Dạy thêm ngữ cảnh (Context learning)
        training_data.append({
            "messages": [
                {"role": "system", "content": "Translate Vietnamese medical abbreviation."},
                {"role": "user", "content": f"Dịch từ viết tắt: Bệnh nhân bị {abbr}"},
                {"role": "assistant", "content": f"Bệnh nhân bị {full}"}
            ]
        })

# Trộn đều
random.shuffle(training_data)

# Lưu file
with open("output/bidirectional_train_data.jsonl", "w", encoding="utf-8") as f:
    for entry in training_data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"✅ Đã tạo data Glossory & Abbreviations. Tổng số mẫu: {len(training_data)}")