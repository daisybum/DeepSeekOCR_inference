#!/usr/bin/env python3
"""
DeepSeek-OCR PDF Inference using Transformers
"""
import os
import io
import fitz  # PyMuPDF
from PIL import Image
from tqdm import tqdm
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import AutoModel, AutoTokenizer

# Configuration
INPUT_PATH = '/home/sanghyun/Projects/DeepSeek-OCR/수지구 임장보고서.pdf'
OUTPUT_PATH = '/home/sanghyun/Projects/DeepSeek-OCR/output'
MODEL_NAME = 'deepseek-ai/DeepSeek-OCR'
PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'

def pdf_to_images(pdf_path, dpi=144):
    """Convert PDF to list of PIL Images"""
    images = []
    pdf_document = fitz.open(pdf_path)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    
    print(f"Converting {pdf_document.page_count} pages...")
    for page_num in tqdm(range(pdf_document.page_count), desc="PDF to Images"):
        page = pdf_document[page_num]
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        img_data = pixmap.tobytes("png")
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        images.append(img)
    
    pdf_document.close()
    return images

def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(f'{OUTPUT_PATH}/images', exist_ok=True)
    
    print(f"Loading PDF: {INPUT_PATH}")
    images = pdf_to_images(INPUT_PATH)
    print(f"Total pages: {len(images)}")
    
    print("\nLoading DeepSeek-OCR model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_NAME, 
        _attn_implementation='flash_attention_2',
        trust_remote_code=True, 
        use_safetensors=True
    )
    model = model.eval().cuda().to(torch.bfloat16)
    print("Model loaded successfully!")
    
    # Process each page
    all_content = []
    for i, img in enumerate(tqdm(images, desc="OCR Processing")):
        # Save temp image
        temp_image_path = f'/tmp/temp_page_{i}.jpg'
        img.save(temp_image_path, 'JPEG', quality=95)
        
        try:
            result = model.infer(
                tokenizer, 
                prompt=PROMPT, 
                image_file=temp_image_path,
                output_path=OUTPUT_PATH,
                base_size=1024,
                image_size=640,
                crop_mode=True,
                save_results=False,
                test_compress=False
            )
            
            if result:
                all_content.append(f"--- Page {i+1} ---\n{result}\n")
            else:
                all_content.append(f"--- Page {i+1} ---\n[OCR failed for this page]\n")
                
        except Exception as e:
            print(f"\nError on page {i+1}: {e}")
            all_content.append(f"--- Page {i+1} ---\n[Error: {str(e)}]\n")
        
        # Clean up temp file
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
    
    # Write output
    output_file = os.path.join(OUTPUT_PATH, '수지구_임장보고서.md')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_content))
    
    print(f"\n✅ OCR completed! Output saved to: {output_file}")

if __name__ == "__main__":
    main()

