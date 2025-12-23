import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ================= CONFIG =================
model_path = "output/merged_qwen_3b"
# ============================================

print("⏳ Loading model...!")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", dtype=torch.float16, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    print("✅ Model được load thành công!")
except Exception as e:
    print(f"❌ Lỗi load model: {e}")
    exit()

# Translate Function (Pipeline Draft -> Refine)
def translate_pipeline(text, direction):
    if not text: return "", ""
    
    # Determine Translation Mode
    if direction == "English -> Vietnamese":
        task_name = "en2vi"
        src_lang = "English"
        tgt_lang = "Vietnamese"
    else:
        task_name = "vi2en"
        src_lang = "Vietnamese"
        tgt_lang = "English"

    # ================= STEP 1: DRAFTING =================
    # Draft Prompt 
    draft_sys = (
        "You are a professional medical translator. Your task is to translate text deeply and accurately."
        "STRICT RULES:\n"
        "1. DO NOT COPY the input. If the input is short, you STILL MUST translate it.\n"
        "2. DO NOT translate proper names (e.g., Vientiane, New York). Keep them original.\n"
        "3. Convert numbers to the target language format (e.g., 12,5% -> 12.5% in English)."
    )
    
    # Zero-shot
    draft_msgs = [
        {"role": "system", "content": draft_sys},
        {"role": "user", "content": f"Translate {src_lang} to {tgt_lang}: {text}"}
    ]
    
    draft_input = tokenizer.apply_chat_template(draft_msgs, tokenize=False, add_generation_prompt=True)
    
    # Inference Draft
    inputs = tokenizer([draft_input], return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=256, 
            use_cache=True, 
            do_sample=False,
            num_beams=4
        )
    draft_result = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    # ================= STEP 2: REFINE =================
    if task_name == "en2vi":
        sys_msg = "You are a senior medical editor. Refine the Vietnamese translation to be more accurate, natural, and use correct medical terminology."
        user_content = f'Original English: "{text}"\nDraft Translation: "{draft_result}"\n\nInstruction: Rewrite it to be perfect.\nRefined Vietnamese:'
        
        # 3-Shot EN->VI
        few_shot = [
            {"role": "user", "content": 'Original English: "The patient was discharged yesterday."\nDraft Translation: "Bệnh nhân đã được xả ngày hôm qua."\n\nInstruction: Rewrite it.\nRefined Vietnamese:'},
            {"role": "assistant", "content": "Bệnh nhân đã được xuất viện ngày hôm qua."},
            {"role": "user", "content": 'Original English: "No significant past medical history."\nDraft Translation: "Không có lịch sử y tế quá khứ đáng kể."\n\nInstruction: Rewrite it.\nRefined Vietnamese:'},
            {"role": "assistant", "content": "Tiền sử bệnh lý không có gì đặc biệt."},
            {"role": "user", "content": 'Original English: "He complained of severe chest pain."\nDraft Translation: "Anh ấy phàn nàn về đau ngực dữ dội."\n\nInstruction: Rewrite it.\nRefined Vietnamese:'},
            {"role": "assistant", "content": "Bệnh nhân than phiền đau ngực dữ dội."}
        ]
        
    else: # vi2en
        sys_msg = "You are a senior medical editor. Refine the English translation to be more accurate, natural, and use correct medical terminology."
        user_content = f'Original Vietnamese: "{text}"\nDraft Translation: "{draft_result}"\n\nInstruction: Rewrite it to be perfect.\nRefined English:'
        
        # 3-Shot VI->EN 
        few_shot = [
            {"role": "user", "content": 'Original Vietnamese: "Bệnh nhân nhập viện vì khó thở."\nDraft Translation: "Patient enter hospital because hard breathe."\n\nInstruction: Rewrite it.\nRefined English:'},
            {"role": "assistant", "content": "The patient was admitted to the hospital due to dyspnea."},
            {"role": "user", "content": 'Original Vietnamese: "Bệnh nhân đã được mổ ruột thừa."\nDraft Translation: "Patient was cut appendix."\n\nInstruction: Rewrite it.\nRefined English:'},
            {"role": "assistant", "content": "The patient underwent an appendectomy."},
            {"role": "user", "content": 'Original Vietnamese: "Kết quả xét nghiệm cho thấy men gan tăng."\nDraft Translation: "Test result show liver enzyme up."\n\nInstruction: Rewrite it.\nRefined English:'},
            {"role": "assistant", "content": "Test results indicated elevated liver enzymes."}
        ]

    refine_msgs = [{"role": "system", "content": sys_msg}] + few_shot + [{"role": "user", "content": user_content}]
    refine_input = tokenizer.apply_chat_template(refine_msgs, tokenize=False, add_generation_prompt=True)

    # Inference Refine
    inputs = tokenizer([refine_input], return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=256, 
            use_cache=True, 
            do_sample=False,
            num_beams=4 
        )
    final_result = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    return draft_result, final_result

# ================= GRADIO UI =================
# Theme
theme = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")

with gr.Blocks(theme=theme, title="Medical Translation Demo") as demo:
    gr.Markdown(
        """
        # 🏥 Medical Translation System (VLSP 2025) by Group 14
        """
    )
    
    with gr.Row():
        # Input
        with gr.Column(scale=1):
            direction = gr.Dropdown(
                choices=["English -> Vietnamese", "Vietnamese -> English"], 
                value="Vietnamese -> English", 
                label="Translation Mode"
            )
            input_text = gr.Textbox(
                lines=5, 
                placeholder="Type medical text here...", 
                label="Input Text"
            )
            
            # Translate Button
            btn_translate = gr.Button("🚀 DỊCH NGAY (TRANSLATE)", variant="primary")
            
            # Examples
            gr.Examples(
                examples=[
                    ["Bệnh nhân nhập viện trong tình trạng đau ngực trái dữ dội, lan lên vai và cánh tay.", "Vietnamese -> English"],
                    ["Kết quả chụp CT cho thấy có khối u ở thùy phổi phải, kích thước 3x4cm.", "Vietnamese -> English"],
                    ["The patient presented with symptoms of acute appendicitis including RLQ pain and fever.", "English -> Vietnamese"],
                    ["Follow-up examination revealed significant improvement in cardiac function.", "English -> Vietnamese"]
                ],
                inputs=[input_text, direction]
            )


        # Output
        with gr.Column(scale=1):
            gr.Markdown("### 🎯 Kết quả dịch (Translation Result)")
            
            # Main Output
            output_final = gr.Textbox(
                label="✨ Final Output (Đã hiệu đính)", 
                interactive=False,
                show_label=True, # Cho nút copy cho tiện
                lines=5
            )
            
            # Draft Step
            with gr.Accordion("🔍 Xem quá trình suy luận (Debug / Draft Step)", open=False):
                output_draft = gr.Textbox(
                    label="📝 Step 1: Draft Translation (Dịch thô ban đầu)", 
                    interactive=False,
                    lines=3
                )
                gr.Markdown(
                    "Note: Hệ thống tự động phát hiện lỗi sai ở bước Draft và sửa lại ở bước Final."
                )

    # Logic
    btn_translate.click(
        fn=translate_pipeline, 
        inputs=[input_text, direction], 
        outputs=[output_draft, output_final]
    )
    
    gr.Markdown("---")
    gr.Markdown("*Developed by ... - Powered by Qwen-3B Finetuned*")

# RUN SERVER
print("🚀 Server đang khởi động...")
demo.launch(share=True, server_port=7860)