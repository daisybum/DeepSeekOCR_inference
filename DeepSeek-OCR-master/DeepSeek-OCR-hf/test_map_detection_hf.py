import torch
from transformers import AutoModel, AutoTokenizer
import os

# ------------------------------------------------------------------
# [설정 영역]
TEST_IMAGE_PATH = "/home/sanghyun/Projects/DeepSeek-OCR/test_map_image.jpg"
MODEL_PATH = "deepseek-ai/DeepSeek-OCR"
OUTPUT_DIR = "/home/sanghyun/Projects/DeepSeek-OCR/output_hf"
# ------------------------------------------------------------------

def run_test():
    print(f"Loading Model: {MODEL_PATH}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        # Disable flash_attention_2 if it causes issues, but try with it first or default
        # Using default attention implementation to avoid potential kernel issues on GB10
        model = AutoModel.from_pretrained(
            MODEL_PATH, 
            trust_remote_code=True, 
            use_safetensors=True
        )
        model = model.eval().cuda().to(torch.bfloat16)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"Error: Image not found at {TEST_IMAGE_PATH}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Scenarios
    scenarios = [
        {
            "name": "Scenario A (Description)",
            "prompt": "<image>\nDescribe this image in detail."
        },
        {
            "name": "Scenario B (Locate Map)",
            "prompt": "<image>\nLocate the map in the image."
        },
        {
            "name": "Scenario C (Markdown)",
            "prompt": "<image>\n<|grounding|>Convert the document to markdown."
        }
    ]

    print("--- Starting Inference (HF) ---")
    for scenario in scenarios:
        print(f"\nRunning {scenario['name']}...")
        try:
            # The model.infer method signature from run_dpsk_ocr.py:
            # infer(self, tokenizer, prompt='', image_file='', output_path = ' ', base_size = 1024, image_size = 640, crop_mode = True, test_compress = False, save_results = False)
            
            # We capture stdout/result if possible, but the method might print or save to file.
            # Based on run_dpsk_ocr.py, it returns 'res'.
            
            res = model.infer(
                tokenizer, 
                prompt=scenario["prompt"], 
                image_file=TEST_IMAGE_PATH, 
                output_path=OUTPUT_DIR, 
                base_size=1024, 
                image_size=640, 
                crop_mode=True, 
                save_results=False, # Don't save images for every step
                test_compress=False
            )
            
            print(f"[{scenario['name']}] Result:\n{res}\n" + "-"*50)
            
        except Exception as e:
            print(f"Error running {scenario['name']}: {e}")

if __name__ == "__main__":
    run_test()
