import os
import json
import mimetypes
from google import genai
from google.genai import types
from dotenv import load_dotenv

# --- 설정 ---
load_dotenv()
API_KEY = os.environ.get('API_KEY') # API 키 입력
TARGET_FILE = "asdf.pdf"      # ★ 여기에 PDF 파일이나 이미지 파일 경로를 넣으세요 (.pdf, .png, .jpg 등)
MODEL_NAME = "gemini-flash-latest" # 회원님 목록에 있는 모델 (또는 gemini-2.0-flash-exp)
# -----------

def analyze_document(file_path, api_key):
    client = genai.Client(api_key=api_key)

    if not os.path.exists(file_path):
        print(f"오류: 파일을 찾을 수 없습니다 -> {file_path}")
        return

    # 1. 파일의 종류(MIME Type)를 자동으로 확인합니다.
    mime_type, _ = mimetypes.guess_type(file_path)
    
    # MIME Type을 못 찾을 경우 확장자로 강제 지정
    if not mime_type:
        if file_path.lower().endswith(".pdf"):
            mime_type = "application/pdf"
        else:
            mime_type = "image/png" # 기본값

    print(f"📂 파일 형식 감지: {mime_type}")

    # 2. 파일을 바이너리(Bytes)로 읽습니다. (PDF, 이미지 공통)
    try:
        with open(file_path, "rb") as f:
            file_data = f.read()
    except Exception as e:
        print(f"파일 읽기 실패: {e}")
        return

    # 3. 프롬프트 작성
    prompt = """
    이 문서를 분석하여 다음 두 가지 작업을 수행하세요.
    
    작업 1: 이 문서가 대한민국 '주택임대차표준계약서' 양식이 맞는지 "예" 또는 "아니오"로 판단하세요.
    작업 2: 만약 "예"라면, '[특약사항]' 란에 적힌 텍스트를 줄바꿈, 빈칸, 체크박스 포함하여 원문 그대로 추출하세요. 
           - 빈칸(____)이나 체크박스(□)도 생략하지 말고 시각적으로 보이는 대로 표현하세요.
           - 문서가 표준계약서가 아니라면 이 항목은 빈 문자열로 두세요.
    """

    print(f"'{MODEL_NAME}' 모델로 분석 중...")
    
    try:
        # 4. API 요청 (contents에 바이트 데이터와 MIME 타입을 직접 전달)
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                types.Part.from_bytes(data=file_data, mime_type=mime_type), # 핵심 변경 부분
                prompt
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema={
                    "type": "OBJECT",
                    "properties": {
                        "is_standard_contract": {"type": "STRING", "enum": ["예", "아니오"]},
                        "special_terms_raw": {"type": "STRING"}
                    }
                }
            )
        )

        return json.loads(response.text)

    except Exception as e:
        print(f"\nAPI 요청 중 오류 발생: {e}")
        return None

# --- 메인 실행 ---
if __name__ == "__main__":
    if API_KEY == "YOUR_GOOGLE_API_KEY":
        print("❌ 오류: API 키를 설정해주세요.")
    else:
        result = analyze_document(TARGET_FILE, API_KEY)

        if result:
            print("\n" + "="*30)
            print("       🔍 분석 결과")
            print("="*30)
            
            is_contract = result.get("is_standard_contract")
            print(f"▶ 표준계약서 여부: {is_contract}")

            if is_contract == "예":
                print("\n▶ 특약사항 원문:")
                print("-" * 30)
                print(result.get("special_terms_raw"))
                print("-" * 30)
            else:
                print("표준계약서 양식이 아닙니다.")