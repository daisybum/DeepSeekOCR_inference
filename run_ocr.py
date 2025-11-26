#!/usr/bin/env python3
"""
DeepSeek-OCR PDF Inference using upstream vLLM
"""
import os
import io
import fitz  # PyMuPDF
from PIL import Image
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from vllm import LLM, SamplingParams

# Check if NGramPerReqLogitsProcessor is available
try:
    from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
    USE_NGRAM_PROCESSOR = True
except ImportError:
    print("Warning: NGramPerReqLogitsProcessor not available, using default logits processors")
    USE_NGRAM_PROCESSOR = False

# Configuration
INPUT_PATH = '/home/sanghyun/Projects/DeepSeek-OCR/수지구 임장보고서.pdf'
OUTPUT_PATH = '/home/sanghyun/Projects/DeepSeek-OCR/output'
MODEL_PATH = 'deepseek-ai/DeepSeek-OCR'
PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'

def pdf_to_images(pdf_path, dpi=144):
    """Convert PDF to list of PIL Images"""
    images = []
    pdf_document = fitz.open(pdf_path)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    
    for page_num in range(pdf_document.page_count):
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
    
    print("Loading DeepSeek-OCR model...")
    
    # Create LLM instance
    llm_kwargs = {
        "model": MODEL_PATH,
        "trust_remote_code": True,
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.9,
        "enable_prefix_caching": False,
        "mm_processor_cache_gb": 0,
    }
    
    if USE_NGRAM_PROCESSOR:
        llm_kwargs["logits_processors"] = [NGramPerReqLogitsProcessor]
    
    llm = LLM(**llm_kwargs)
    
    # Prepare inputs
    print("Preparing inputs...")
    model_inputs = []
    for img in tqdm(images, desc="Processing pages"):
        model_inputs.append({
            "prompt": PROMPT,
            "multi_modal_data": {"image": img}
        })
    
    # Sampling parameters
    sampling_kwargs = {
        "temperature": 0.0,
        "max_tokens": 8192,
        "skip_special_tokens": False,
    }
    
    if USE_NGRAM_PROCESSOR:
        sampling_kwargs["extra_args"] = {
            "ngram_size": 30,
            "window_size": 90,
            "whitelist_token_ids": {128821, 128822},  # <td>, </td>
        }
    
    sampling_params = SamplingParams(**sampling_kwargs)
    
    # Generate
    print("Running OCR inference...")
    outputs = llm.generate(model_inputs, sampling_params)
    
    # Save results
    all_content = []
    for i, output in enumerate(outputs):
        content = output.outputs[0].text
        # Clean up end token if present
        if '<｜end▁of▁sentence｜>' in content:
            content = content.replace('<｜end▁of▁sentence｜>', '')
        all_content.append(f"--- Page {i+1} ---\n{content}\n")
    
    # Write output
    output_file = os.path.join(OUTPUT_PATH, '수지구_임장보고서.md')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_content))
    
    print(f"\nOCR completed! Output saved to: {output_file}")

if __name__ == "__main__":
    main()

