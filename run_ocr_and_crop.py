import os
import sys
import fitz
import io
import re
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch

if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"
else:
    # Try to find ptxas in torch bin
    torch_ptxas = os.path.join(os.path.dirname(torch.__file__), 'bin', 'ptxas')
    if os.path.exists(torch_ptxas):
        os.environ["TRITON_PTXAS_PATH"] = torch_ptxas
        print(f"Using torch ptxas at {torch_ptxas}")

os.environ['VLLM_USE_V1'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Use vLLM imports as requested
from vllm import LLM, SamplingParams
try:
    from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
except ImportError:
    print("Warning: NGramPerReqLogitsProcessor not found in vllm.model_executor.models.deepseek_ocr, trying default")
    NGramPerReqLogitsProcessor = None

# Configuration
INPUT_PATH = '/home/sanghyun/Projects/DeepSeek-OCR/input/수지구 임장보고서.pdf'
OUTPUT_PATH = '/home/sanghyun/Projects/DeepSeek-OCR/output'
MODEL_PATH = 'deepseek-ai/DeepSeek-OCR'
PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'
PADDING = 10  # Padding for cropping

def pdf_to_images(pdf_path, dpi=144):
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

def extract_coordinates_and_label(ref_text):
    try:
        # Expected format: <|ref|>label<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>
        # Regex in re_match handles extraction, here we parse the inner parts
        # But re_match returns full strings. Let's parse them.
        # Actually, the previous code passed (label_type, cor_list) tuple.
        # We need to parse the string to get that.
        
        # ref_text is like: <|ref|>title<|/ref|><|det|>[[310, 355, 685, 460]]<|/det|>
        label_match = re.search(r'<\|ref\|>(.*?)<\|/ref\|>', ref_text)
        det_match = re.search(r'<\|det\|>\[\[(.*?)\]\]<\|/det\|>', ref_text)
        
        if label_match and det_match:
            label_type = label_match.group(1)
            coords_str = det_match.group(1)
            coords = [int(x.strip()) for x in coords_str.split(',')]
            return label_type, [coords]
    except Exception as e:
        print(f"Error parsing ref: {e}")
    return None

def re_match(text):
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)
    
    # matches is list of tuples: (full_match, label, det_content)
    # We want list of full matches
    full_matches = [m[0] for m in matches]
    
    matches_image = []
    matches_other = []
    for m in full_matches:
        if '<|ref|>image<|/ref|>' in m:
            matches_image.append(m)
        else:
            matches_other.append(m)
            
    return full_matches, matches_image, matches_other

def draw_bounding_boxes(image, refs, jdx):
    image_width, image_height = image.size
    img_idx = 0
    
    for ref in refs:
        result = extract_coordinates_and_label(ref)
        if result:
            label_type, points_list = result
            
            for points in points_list:
                x1_norm, y1_norm, x2_norm, y2_norm = points

                # Apply padding and clamp
                x1_norm = max(0, x1_norm - PADDING)
                y1_norm = max(0, y1_norm - PADDING)
                x2_norm = min(1000, x2_norm + PADDING)
                y2_norm = min(1000, y2_norm + PADDING)

                x1 = int(x1_norm / 1000 * image_width)
                y1 = int(y1_norm / 1000 * image_height)
                x2 = int(x2_norm / 1000 * image_width)
                y2 = int(y2_norm / 1000 * image_height)

                if label_type == 'image':
                    try:
                        cropped = image.crop((x1, y1, x2, y2))
                        save_path = f"{OUTPUT_PATH}/images/{jdx}_{img_idx}.jpg"
                        cropped.save(save_path)
                        img_idx += 1
                    except Exception as e:
                        print(f"Error cropping: {e}")

def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(f'{OUTPUT_PATH}/images', exist_ok=True)
    
    print("Loading PDF...")
    images = pdf_to_images(INPUT_PATH)
    
    print("Initializing Model...")
    logits_processors = []
    if NGramPerReqLogitsProcessor:
        logits_processors = [NGramPerReqLogitsProcessor]

    llm = LLM(
        model=MODEL_PATH,
        enable_prefix_caching=False,
        mm_processor_cache_gb=0,
        logits_processors=logits_processors,
        max_num_seqs=100, # Concurrency
        gpu_memory_utilization=0.8,
        trust_remote_code=True
    )
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        skip_special_tokens=False,
        # extra_args for NGramPerReqLogitsProcessor if used
    )
    
    # If NGramPerReqLogitsProcessor is used, we might need to pass extra_args in sampling_params
    # The user's snippet shows extra_args in SamplingParams.
    if NGramPerReqLogitsProcessor:
         sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=8192,
            extra_args=dict(
                ngram_size=20,
                window_size=50,
                whitelist_token_ids={128821, 128822},
            ),
            skip_special_tokens=False,
        )

    print("Preparing inputs...")
    batch_inputs = []
    for img in images:
        batch_inputs.append({
            "prompt": PROMPT,
            "multi_modal_data": {"image": img}
        })
        
    print("Generating...")
    outputs_list = llm.generate(batch_inputs, sampling_params)
    
    print("Processing outputs...")
    mmd_path = os.path.join(OUTPUT_PATH, '수지구_임장보고서_ocr_vllm.md')
    
    full_content = ""
    
    for jdx, (output, img) in enumerate(zip(outputs_list, images)):
        content = output.outputs[0].text
        
        # Parse and crop
        matches_ref, matches_image, matches_other = re_match(content)
        draw_bounding_boxes(img, matches_ref, jdx)
        
        # Replace image refs with paths
        img_idx = 0
        for m in matches_image:
            # We assume the order matches the cropping order
            # This might be fragile if re_match order differs from draw_bounding_boxes iteration
            # But draw_bounding_boxes iterates matches_ref which includes matches_image
            # Let's just replace blindly for now or try to match index
            content = content.replace(m, f'![](images/{jdx}_{img_idx}.jpg)\n')
            img_idx += 1
            
        full_content += f"## Page {jdx+1}\n\n{content}\n\n"
        
    with open(mmd_path, 'w', encoding='utf-8') as f:
        f.write(full_content)
        
    print(f"Done. Markdown saved to {mmd_path}")

if __name__ == "__main__":
    main()
