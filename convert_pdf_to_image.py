import fitz  # PyMuPDF
import sys

def convert_pdf_to_image(pdf_path, output_path, page_num=0):
    try:
        doc = fitz.open(pdf_path)
        if page_num >= len(doc):
            print(f"Error: Page {page_num} out of range (max {len(doc)-1})")
            return
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        pix.save(output_path)
        print(f"Successfully converted {pdf_path} page {page_num} to {output_path}")
    except Exception as e:
        print(f"Error converting PDF: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Hardcoded paths as per user context
    pdf_path = "/home/sanghyun/Projects/DeepSeek-OCR/input/수지구 임장보고서.pdf"
    
    # Convert page 12 (index 12) which contains "지도"
    output_path = "/home/sanghyun/Projects/DeepSeek-OCR/test_map_image.jpg"
    convert_pdf_to_image(pdf_path, output_path, page_num=12)
