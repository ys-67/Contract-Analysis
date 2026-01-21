import os
from google import genai
from dotenv import load_dotenv

# API 키 입력
load_dotenv()
API_KEY = os.environ.get('API_KEY')

def list_available_models():
    # 클라이언트 초기화
    client = genai.Client(api_key=API_KEY)
    
    print("🔍 모델 목록 조회 중...\n")
    try:
        # 조건문 없이 모든 모델 이름 출력
        for model in client.models.list():
            # 모델 이름 부분만 잘라서 출력 (예: models/gemini-1.5-flash -> gemini-1.5-flash)
            if hasattr(model, 'name'):
                print(f"- {model.name.replace('models/', '')}")
            else:
                print(f"- {model}") # 이름 속성이 없으면 전체 출력
                
    except Exception as e:
        print(f"목록 조회 실패: {e}")

if __name__ == "__main__":
    list_available_models()