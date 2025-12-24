import re

# ========= CONFIG =========
TRAIN_VI = "input/train.vi.txt"
TRAIN_EN = "input/train.en.txt"
TEST_VI  = "input/public_test.vi.txt"
TEST_EN  = "input/public_test.en.txt"

OUT_VI = "input/clean_train.vi.txt"
OUT_EN = "input/clean_train.en.txt"
# ==========================


def normalize_numbers(text: str) -> str:
    """
    Chuẩn hoá:
    - 4,2%  -> 4.2%
    - 4, 2% -> 4.2%
    """
    
    text = re.sub(r"(\d+)\s*,\s*(\d+)\s*%", r"\1.\2%", text)
    return text


def clean_line(line: str) -> str:
    line = line.strip()
    line = normalize_numbers(line)
    return line


def load_parallel(vi_path, en_path):
    with open(vi_path, encoding="utf-8") as f:
        vi_lines = f.readlines()
    with open(en_path, encoding="utf-8") as f:
        en_lines = f.readlines()

    assert len(vi_lines) == len(en_lines), "VI–EN không cùng số dòng"

    pairs = []
    for vi, en in zip(vi_lines, en_lines):
        vi = clean_line(vi)
        en = clean_line(en)
        pairs.append((vi, en))

    return pairs


def main():
    print("🔹 Load test set...")
    test_pairs = set(load_parallel(TEST_VI, TEST_EN))

    print("🔹 Load train set...")
    train_pairs = load_parallel(TRAIN_VI, TRAIN_EN)

    print(f"📊 Train ban đầu: {len(train_pairs)}")

    # Lọc train trùng test
    cleaned_pairs = [
        (vi, en)
        for (vi, en) in train_pairs
        if (vi, en) not in test_pairs
    ]

    print(f"✅ Sau khi loại test overlap: {len(cleaned_pairs)}")
    print(f"❌ Bị loại: {len(train_pairs) - len(cleaned_pairs)}")

    # Ghi file
    with open(OUT_VI, "w", encoding="utf-8") as f_vi, \
         open(OUT_EN, "w", encoding="utf-8") as f_en:
        for vi, en in cleaned_pairs:
            f_vi.write(vi + "\n")
            f_en.write(en + "\n")

    print("🎉 XONG! Đã ghi file:")
    print("   -", OUT_VI)
    print("   -", OUT_EN)


if __name__ == "__main__":
    main()
