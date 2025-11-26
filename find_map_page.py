import fitz

def find_map_page(pdf_path):
    doc = fitz.open(pdf_path)
    keywords = ["지도", "위치도", "현황도", "Map", "Location"]
    
    print(f"Scanning {len(doc)} pages for keywords: {keywords}")
    
    for i, page in enumerate(doc):
        text = page.get_text()
        for kw in keywords:
            if kw in text:
                print(f"Found '{kw}' on page {i} (index {i})")
                # Print a snippet
                print(f"  Context: {text[:100].replace('\n', ' ')}...")
                return i
    
    print("No map keywords found.")
    return -1

if __name__ == "__main__":
    pdf_path = "/home/sanghyun/Projects/DeepSeek-OCR/input/수지구 임장보고서.pdf"
    find_map_page(pdf_path)
