from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import pandas as pd
import random
import sacrebleu
import os

print("🚀 RUNNING FULL PIPELINE: DRAFT -> REFINE...")

# ================= CONFIG =================
model_path = "output/merged_qwen_3b"
BATCH_SIZE = 32        
SAMPLE_SIZE = 3000     # Number of test samples

tasks = [
    { 
        "name": "EN -> VI", 
        "src_file": "input/public_test.en.txt", 
        "tgt_file": "input/public_test.vi.txt", 
        "output_csv": "output/pipeline_result_en2vi.csv",
        "mode": 1 
    },
    { 
        "name": "VI -> EN", 
        "src_file": "input/public_test.vi.txt", 
        "tgt_file": "input/public_test.en.txt",
        "output_csv": "output/pipeline_result_vi2en.csv",
        "mode": 2 
    }
]
# ============================================

# Load Model
print(f"⏳ Loading model from {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=True, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", dtype=torch.float16, trust_remote_code=True)

tokenizer.padding_side = "left"
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

# Generate Function
def run_generate(prompts):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            use_cache=True,
            num_beams=4, 
            early_stopping=True,
            do_sample=False,
            repetition_penalty=1.2,
            length_penalty=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    decoded = tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return [p.strip() for p in decoded]

# ================= MAIN =================
for task in tasks:
    print(f"\n" + "="*50)
    print(f"🔥 PROCESSING TASK: {task['name']} | Sample: {SAMPLE_SIZE}")
    print("="*50)
    
    mode = task['mode']
    src_lang = "English" if mode == 1 else "Vietnamese"
    tgt_lang = "Vietnamese" if mode == 1 else "English"
    
    # 1. Reading Input Files
    with open(task['src_file'], "r", encoding="utf-8") as f: src_lines = [l.strip() for l in f]
    with open(task['tgt_file'], "r", encoding="utf-8") as f: tgt_lines = [l.strip() for l in f]
    
    # 2. Random Sampling
    all_indices = list(range(len(src_lines)))
    random.seed(42)
    sampled_indices = random.sample(all_indices, SAMPLE_SIZE)
    
    data = []
    for idx in sampled_indices:
        data.append({
            "id": idx,
            "source": src_lines[idx],
            "target": tgt_lines[idx]
        })
        
    # 3. Smart Batching (Sort by source sentence length)
    print("⏳ Đang sắp xếp dữ liệu (Smart Batching)...")
    data_sorted = sorted(data, key=lambda x: len(x["source"]))
    
    results = []
    
    # 4. Batch Running (Draft -> Refine)
    total_batches = len(data_sorted) // BATCH_SIZE + 1
    
    for i in tqdm(range(0, len(data_sorted), BATCH_SIZE), total=total_batches):
        batch = data_sorted[i : i + BATCH_SIZE]
        if not batch: continue
        
        # === PHASE 1: DRAFTING ===
        draft_prompts = []
        for item in batch:
            sentence = item["source"]
            # Draft Prompt
            sys_prompt = (
                  "You are a professional medical translator. Your task is to translate text deeply and accurately."
                  "STRICT RULES:\n"
                  "1. DO NOT COPY the input. If the input is short, you STILL MUST translate it.\n"
                  "2. DO NOT translate proper names (e.g., Vientiane, New York). Keep them original.\n"
                  "3. Convert numbers to the target language format (e.g., 12,5% -> 12.5% in English)."
            )
            
            few_shot = [
                {"role": "system", "content":  sys_prompt},
                
                # Remove few-shot since zero-shot provides better score in both BLEU and LLM Score
                
#                # Translate Short Sentence
#                {"role": "user", "content": "Translate the following medical sentence from English to Vietnamese accurately: Hello"},
#                {"role": "assistant", "content": "Xin chào"},
#                    
#                {"role": "user", "content": "Translate the following medical sentence from Vietnamese to English accurately: Không"},
#                {"role": "assistant", "content": "No"},
#                
#                # Translate with correct decimal format
#                {"role": "user", "content": "Translate the following medical sentence from English to Vietnamese accurately: The mortality rate decreased to 12.5%."},
#                {"role": "assistant", "content": "Tỷ lệ tử vong giảm xuống còn 12,5%."},
#                
#                {"role": "user", "content": "Translate the following medical sentence from Vietnamese to English accurately: Tỷ lệ tử vong giảm xuống còn 12,5%."},
#                {"role": "assistant", "content": "The mortality rate decreased to 12.5%."},
#                
#                # E.g. 1: Keep "New York"
#                {"role": "user", "content": f"Translate the following medical sentence from English to Vietnamese accurately: The patient was transferred to a hospital in New York."},
#                {"role": "assistant", "content": "Bệnh nhân đã được chuyển đến một bệnh viện tại New York."},
#                
##                # E.g. 2: Keep "Paris"
#                {"role": "user", "content": f"Translate the following medical sentence from English to Vietnamese accurately: Research conducted in Paris shows significant results."},
#                {"role": "assistant", "content": "Nghiên cứu được thực hiện tại Paris cho thấy kết quả đáng kể."},
#                
#                # E.g. 3: Keep "Nghệ An"
#                {"role": "user", "content": f"Translate the following medical sentence from Vietnamese to English accurately: Bệnh nhân bị VXCC ở Nghệ An."},
#                {"role": "assistant", "content": "The patient has acute mastoiditis at Nghe An."},
#                
#                # E.g. 4: Keep "Hà Nội"
#                {"role": "user", "content": f"Translate the following medical sentence from Vietnamese to English accurately: Mô tả đặc điểm lâm sàng, cận lâm sàng, thực trạng điều trị bệnh nhân suy tim cấp nhập viện tại Bệnh viện Tim Hà Nội"},
#                {"role": "assistant", "content": "Clinical, paraclinical characteristics and real-world treatment of acute heart failure at Hanoi heart hospital"},
                
                # Command for translation task
                {"role": "user", "content": f"Translate the following medical sentence from {src_lang} to {tgt_lang} accurately: {sentence}"}
            ]
            draft_prompts.append(tokenizer.apply_chat_template(few_shot, tokenize=False, add_generation_prompt=True))
            
        # Generate Draft
        draft_outputs = run_generate(draft_prompts)
        
        # === PHASE 2: REFINING ===
        refine_prompts = []
        for item, draft in zip(batch, draft_outputs):
            src = item['source']
            
            # Refine Prompt
            if mode == 2: # VI -> EN
                sys_msg = "You are a senior medical editor. Refine the English translation to be more accurate, natural, and use correct medical terminology."
                user_content = f'Original Vietnamese: "{src}"\nDraft Translation: "{draft}"\n\nInstruction: Rewrite it to be perfect.\nRefined English:'
                # 3-Shot VI->EN 
                few_shot = [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": 'Original Vietnamese: "Bệnh nhân nhập viện vì khó thở."\nDraft Translation: "Patient enter hospital because hard breathe."\n\nInstruction: Rewrite it.\nRefined English:'},
                    {"role": "assistant", "content": "The patient was admitted to the hospital due to dyspnea."},
                    {"role": "user", "content": 'Original Vietnamese: "Bệnh nhân đã được mổ ruột thừa."\nDraft Translation: "Patient was cut appendix."\n\nInstruction: Rewrite it.\nRefined English:'},
                    {"role": "assistant", "content": "The patient underwent an appendectomy."},
                    {"role": "user", "content": user_content}
                ]
            else: # EN -> VI
                sys_msg = "You are a senior medical editor. Refine the Vietnamese translation to be more accurate, natural, and use correct medical terminology."
                user_content = f'Original English: "{src}"\nDraft Translation: "{draft}"\n\nInstruction: Rewrite it to be perfect.\nRefined Vietnamese:'
                # 3-Shot EN->VI 
                few_shot = [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": 'Original English: "The patient was discharged yesterday."\nDraft Translation: "Bệnh nhân đã được xả ngày hôm qua."\n\nInstruction: Rewrite it.\nRefined Vietnamese:'},
                    {"role": "assistant", "content": "Bệnh nhân đã được xuất viện ngày hôm qua."},
                    {"role": "user", "content": 'Original English: "No significant past medical history."\nDraft Translation: "Không có lịch sử y tế quá khứ đáng kể."\n\nInstruction: Rewrite it.\nRefined Vietnamese:'},
                    {"role": "assistant", "content": "Tiền sử bệnh lý không có gì đặc biệt."},
                    {"role": "user", "content": user_content}
                ]

            refine_prompts.append(tokenizer.apply_chat_template(few_shot, tokenize=False, add_generation_prompt=True))
        
        # Generate Refine
        refine_outputs = run_generate(refine_prompts)
        
        # Save Result (list)
        for item, draft, final in zip(batch, draft_outputs, refine_outputs):
            results.append({
                "id": item['id'],
                "source": item['source'],
                "target": item['target'],
                "draft_prediction": draft,
                "refined_prediction": final
            })

    # 5. Sort result by id and save .csv
    results.sort(key=lambda x: x['id'])
    df = pd.DataFrame(results)
    
    print(f"💾 Saving pipeline result to {task['output_csv']}...")
    df.to_csv(task['output_csv'], index=False, encoding="utf-8-sig")
    
    # 6. Calculate BLEU
    print(f"\n📊 KẾT QUẢ DRAFT ({task['name']}):")
    refs = [[str(r).strip() for r in df["target"].tolist()]]
    
    # Draft Score
    drafts = [str(p).strip() for p in df["draft_prediction"].tolist()]
    bleu_draft = sacrebleu.corpus_bleu(drafts, refs, tokenize='13a')
    print(f"🔹 BLEU Draft:  {bleu_draft.score:.2f}")
    
    # Refine Score
    finals = [str(p).strip() for p in df["refined_prediction"].tolist()]
    bleu_final = sacrebleu.corpus_bleu(finals, refs, tokenize='13a')
    print(f"🔸 BLEU Refine: {bleu_final.score:.2f}")
    
    if bleu_final.score > bleu_draft.score:
        print("🚀 Refine hiệu quả! Điểm tăng!")
    else:
        print("⚠️ Refine không tăng điểm.")

print("\n🎉 ALL DONE! Pipeline chạy không lỗi!")