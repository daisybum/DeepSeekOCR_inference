import os
from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
from PIL import Image

# ------------------------------------------------------------------
# [설정 영역] 테스트할 이미지 경로를 입력하세요.
TEST_IMAGE_PATH = "/home/sanghyun/Projects/DeepSeek-OCR/test_map_image.jpg"
MODEL_PATH = "deepseek-ai/DeepSeek-OCR"
# ------------------------------------------------------------------

def run_test():
    # 1. 모델 로드 (vLLM)
    print(f"Loading Model: {MODEL_PATH}...")
    try:
        llm = LLM(
            model=MODEL_PATH,
            enable_prefix_caching=False,
            mm_processor_cache_gb=0,
            logits_processors=[NGramPerReqLogitsProcessor],
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. 이미지 준비
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"Error: Image not found at {TEST_IMAGE_PATH}")
        return
    
    try:
        image = Image.open(TEST_IMAGE_PATH).convert("RGB")
    except Exception as e:
        print(f"Error opening image: {e}")
        return

    # 3. 테스트할 프롬프트 시나리오 준비
    prompt_desc = "<image>\nDescribe this image in detail."
    prompt_locate = "<image>\nLocate the map in the image."
    prompt_md = "<image>\n<|grounding|>Convert the document to markdown."

    prompts = [
        {"prompt": prompt_desc, "multi_modal_data": {"image": image}, "name": "Scenario A (Description)"},
        {"prompt": prompt_locate, "multi_modal_data": {"image": image}, "name": "Scenario B (Locate Map)"},
        {"prompt": prompt_md, "multi_modal_data": {"image": image}, "name": "Scenario C (Markdown)"},
    ]

    # 4. 파라미터 설정
    sampling_param = SamplingParams(
        temperature=0.0,
        max_tokens=2048,
        extra_args=dict(
            ngram_size=30,
            window_size=90,
            whitelist_token_ids={128821, 128822}, # <td>, </td>
        ),
        skip_special_tokens=False,
    )

    # 5. 추론 및 결과 출력
    print("--- Starting Inference ---")
    try:
        outputs = llm.generate([p["prompt"] for p in prompts], sampling_param)

        print("\n" + "="*50)
        for i, output in enumerate(outputs):
            scenario_name = prompts[i]["name"]
            generated_text = output.outputs[0].text
            print(f"[{scenario_name}] Result:\n")
            print(generated_text)
            print("-" * 50)
    except Exception as e:
        print(f"Error during inference: {e}")

if __name__ == "__main__":
    run_test()
