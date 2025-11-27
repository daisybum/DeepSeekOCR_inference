import os
import re
import fitz  # PyMuPDF
from PIL import Image
import io
from tqdm import tqdm

# Configuration
PDF_PATH = '/home/sanghyun/Projects/DeepSeek-OCR/input/수지구 임장보고서.pdf'
OCR_MD_PATH = '/home/sanghyun/Projects/DeepSeek-OCR/output/수지구_임장보고서_ocr.md'
OUTPUT_DIR = '/home/sanghyun/Projects/DeepSeek-OCR/output/crops'
PADDING = 10  # Padding in normalized 0-1000 units

def pdf_to_images(pdf_path, dpi=144):
    """Convert PDF to list of PIL Images"""
    images = []
    try:
        pdf_document = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF: {e}")
        return []

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

def parse_markdown_and_crop(md_path, images, output_dir):
    """Parse markdown for bounding boxes and crop images"""
    
    if not os.path.exists(md_path):
        print(f"Error: Markdown file not found at {md_path}")
        return

    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by pages
    # Assuming format "## Page X"
    pages = re.split(r'## Page (\d+)', content)
    
    # pages[0] is usually empty or header
    # pages[1] is page num, pages[2] is content, pages[3] is page num, pages[4] is content...
    
    total_crops = 0
    
    # Iterate through pages
    # We start from index 1 because split puts the first chunk (before first match) at 0
    for i in range(1, len(pages), 2):
        page_num_str = pages[i]
        page_content = pages[i+1]
        
        try:
            page_idx = int(page_num_str) - 1 # 0-indexed
        except ValueError:
            print(f"Skipping invalid page number: {page_num_str}")
            continue
            
        if page_idx >= len(images):
            print(f"Warning: Page {page_idx+1} in markdown exceeds PDF page count {len(images)}")
            continue
            
        image = images[page_idx]
        width, height = image.size
        
        # Find all bounding boxes and their preceding refs
        # Pattern: <|ref|>label<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>
        # Also handle cases where there might be text between ref and det, or just det
        # But looking at the user provided snippet:
        # <|ref|>title<|/ref|><|det|>[[310, 355, 685, 460]]<|/det|>
        # <|ref|>text<|/ref|><|det|>[[405, 496, 591, 525]]<|/det|>
        
        # We will look for the specific pattern <|det|>[[...]]<|/det|> and try to find the tag before it.
        
        # Regex to find all dets and optionally the ref tag before it
        # We search for <|det|> first, then look back? Or just search for the pair?
        # Sometimes there might be no ref?
        
        # Let's iterate through all <|det|> tags
        det_pattern = r"(?:<\|ref\|>(.*?)<\|/ref\|>)?\s*<\|det\|>\[\[(.*?)\]\]<\|/det\|>"
        
        matches = re.finditer(det_pattern, page_content)
        
        page_crop_count = 0
        for match in matches:
            label = match.group(1)
            bbox_str = match.group(2)
            
            if not label:
                label = "unknown"
            
            try:
                coords = [int(x.strip()) for x in bbox_str.split(',')]
                if len(coords) != 4:
                    continue
                
                x1_norm, y1_norm, x2_norm, y2_norm = coords
                
                # Denormalize with padding
                x1_norm = max(0, x1_norm - PADDING)
                y1_norm = max(0, y1_norm - PADDING)
                x2_norm = min(1000, x2_norm + PADDING)
                y2_norm = min(1000, y2_norm + PADDING)

                x1 = int(x1_norm / 1000 * width)
                y1 = int(y1_norm / 1000 * height)
                x2 = int(x2_norm / 1000 * width)
                y2 = int(y2_norm / 1000 * height)
                
                # Clamp
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Crop
                crop = image.crop((x1, y1, x2, y2))
                
                # Filename: page_{page_num}_{label}_{index}.jpg
                # Sanitize label
                safe_label = "".join([c if c.isalnum() else "_" for c in label])
                filename = f"page_{page_idx+1}_{safe_label}_{page_crop_count}.jpg"
                save_path = os.path.join(output_dir, filename)
                
                crop.save(save_path)
                page_crop_count += 1
                total_crops += 1
                
            except ValueError:
                continue
        
        # print(f"Page {page_idx+1}: Saved {page_crop_count} crops")

    print(f"\n✅ Total crops saved: {total_crops}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading PDF: {PDF_PATH}")
    images = pdf_to_images(PDF_PATH)
    
    if not images:
        print("Failed to load PDF images.")
        return
        
    print(f"Parsing Markdown: {OCR_MD_PATH}")
    parse_markdown_and_crop(OCR_MD_PATH, images, OUTPUT_DIR)
    
    print(f"Done. Crops saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
