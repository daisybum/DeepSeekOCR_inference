#!/usr/bin/env python3
"""
Commercial Location Analysis using DeepSeek-OCR and LangChain Prompts
"""
import os
import io
import fitz  # PyMuPDF
from PIL import Image
from tqdm import tqdm
from prompts import PromptManager

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from vllm import LLM, SamplingParams

# Check if NGramPerReqLogitsProcessor is available (upstream vLLM)
try:
    from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
    USE_NGRAM_PROCESSOR = True
    print("✅ NGramPerReqLogitsProcessor available (upstream vLLM)")
except ImportError:
    print("⚠️ NGramPerReqLogitsProcessor not available, using default logits processors")
    USE_NGRAM_PROCESSOR = False

# Configuration
INPUT_PATH = '/home/sanghyun/Projects/DeepSeek-OCR/input/수지구 임장보고서.pdf'
OUTPUT_PATH = '/home/sanghyun/Projects/DeepSeek-OCR/output'
MODEL_PATH = 'deepseek-ai/DeepSeek-OCR'

def pdf_to_images(pdf_path, dpi=144):
    """Convert PDF to list of PIL Images"""
    images = []
    pdf_document = fitz.open(pdf_path)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    
    print(f"📄 Converting {pdf_document.page_count} pages to images...")
    for page_num in tqdm(range(pdf_document.page_count), desc="PDF → Images"):
        page = pdf_document[page_num]
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        img_data = pixmap.tobytes("png")
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        images.append(img)
    
    pdf_document.close()
    return images

def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    print(f"\n📥 Loading PDF: {INPUT_PATH}")
    images = pdf_to_images(INPUT_PATH)
    print(f"✅ Total pages: {len(images)}")
    
    print("\n🔧 Loading DeepSeek-OCR model...")
    
    # Create LLM instance
    llm_kwargs = {
        "model": MODEL_PATH,
        "enable_prefix_caching": False,
        "mm_processor_cache_gb": 0,
    }
    
    if USE_NGRAM_PROCESSOR:
        llm_kwargs["logits_processors"] = [NGramPerReqLogitsProcessor]
    
    llm = LLM(**llm_kwargs)
    print("✅ Model loaded successfully!")
    
    # --- PROMPT GENERATION WITH LANGCHAIN ---
    print("\n🧠 Generating Prompt with LangChain...")
    prompt_template = PromptManager.get_location_analysis_prompt()
    # We pass the specific instruction as 'user_input'
    final_prompt = prompt_template.format(user_input="Analyze the commercial potential of the location shown in the image.")
    
    print(f"📝 Final Prompt Preview:\n{'-'*40}\n{final_prompt}\n{'-'*40}")
    
    # Prepare batched input
    model_inputs = []
    for img in images:
        model_inputs.append({
            "prompt": final_prompt,
            "multi_modal_data": {"image": img}
        })
    
    # Sampling parameters
    # DeepSeek R1/Reasoning style recommendation: Temperature ~0.6
    sampling_kwargs = {
        "temperature": 0.6, 
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
    
    # Generate output
    print("\n🚀 Running Analysis Inference...")
    model_outputs = llm.generate(model_inputs, sampling_params)
    
    # Process and save results
    all_content = []
    for i, output in enumerate(model_outputs):
        content = output.outputs[0].text
        # Clean up end token if present
        if '<｜end of sentence｜>' in content:
            content = content.replace('<｜end of sentence｜>', '')
            
        # Add page header
        page_result = f"## Page {i+1} Analysis\n\n{content}\n"
        all_content.append(page_result)
        print(f"  Page {i+1} analyzed.")
    
    # Write output
    output_file = os.path.join(OUTPUT_PATH, '수지구_임장보고서_analysis.md')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# Commercial Location Analysis Report\n\n")
        f.write(f"**Source Document**: {os.path.basename(INPUT_PATH)}\n")
        f.write(f"**Model**: {MODEL_PATH}\n")
        f.write(f"**Prompt Strategy**: LangChain Few-Shot\n\n")
        f.write('---\n\n'.join(all_content))
    
    print(f"\n✅ Analysis completed! Report saved to: {output_file}")

if __name__ == "__main__":
    main()
