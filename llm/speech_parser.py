import openai
import json
import re
import os
from datetime import datetime
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 환경 변수에서 API 키 가져오기
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI API 키 설정
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# clothing_item 범주 정의
CATEGORIES = [
    "Short Sl. Shirt",
    "Long Sl. Shirt",
    "Short Sl. Outw.",
    "Long Sl. Outw.",
    "Vest",
    "Sling",
    "Shorts",
    "Trousers",
    "Skirt",
    "Short Sl. Dress",
    "Long Sl. Dress",
    "Vest Dress",
    "Sling Dress"
]

COLORS = [
    'White', 'Off-White', 'Ivory', 'Cream', 'Beige', 'Black', 'Charcoal',
    'Gray', 'Light Gray', 'Silver', 'Red', 'Crimson', 'Burgundy', 'Maroon',
    'Scarlet', 'Brick Red', 'Orange', 'Tangerine', 'Peach', 'Coral', 'Yellow',
    'Mustard', 'Gold', 'Green', 'Lime', 'Olive', 'Forest Green', 'Emerald',
    'Teal', 'Blue', 'Sky Blue', 'Navy', 'Royal Blue', 'Cobalt', 'Turquoise',
    'Sapphire', 'Denim', 'Purple', 'Lavender', 'Violet', 'Pink', 'Hot Pink',
    'Blush', 'Rose', 'Magenta', 'Salmon', 'Brown', 'Tan', 'Mocha', 'Chocolate',
    'Bronze'
]

# 테스트를 위한 임시 user_sentence 값
user_sentence = "차은우가 입고 있는 쨍한 핑크 가디건 마음에 들어"

def parse_speech_to_json(speech_text):
    """GPT를 활용하여 유저 발화 내용을 JSON 형태로 변환"""

    # 프롬프트 작성 (카테고리 내에서만 선택하도록 GPT에게 명확히 지시)
    prompt = f"""
    TV에 등장하는 의상에 대해 간단히 설명하는 문장을 제공할 것입니다. 
    이를 JSON 형식으로 변환해주세요. 
    지시 사항
    1. 의류 종류(`clothing_item`)는 아래 리스트에서만 선택 가능합니다. (단 반드시 하나의 CATEGORIES를 골라야만 합니다.):
    
    가능 항목:
    {CATEGORIES}
    
    2. 의류 색상(`color`)는 아래 리스트에서만 선택 가능합니다. (단 반드시 하나의 COLORS를 골라야만 합니다.):
    
    가능 항목:
    {COLORS}

    JSON 구조는 다음과 같습니다:
    
    {{
        "features": {{
            "clothing_item": "CATEGORIES 항목 중 하나 (반드시 CATEGORIES 중 하나를 골라야만 함, 없으면 null)",
            "color": "COLORS 항목 중 하나 (반드시 COLORS 중 하나를 골라야만 함, 없으면 null)",
            "location": "Left, Right 등 위치 (없으면 null)"
        }}
    }}
    
    아래 문장을 위 JSON 형식으로 변환해주세요:
    "{speech_text}"
    """

    # GPT API 요청 (최신 방식)
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "당신은 텍스트 문장을 JSON으로 변환하는 도우미입니다."},
                      {"role": "user", "content": prompt}]
        )

        # GPT 응답 내용 확인
        structured_json = response.choices[0].message.content.strip()

        # 응답이 비어 있거나 예상과 다를 경우 예외 처리
        if not structured_json:
            raise ValueError("GPT 응답이 비어 있습니다.")

        # 백틱(````json` ... ` ``` ```) 제거
        structured_json = re.sub(r"```json\n(.*?)\n```", r"\1", structured_json, flags=re.DOTALL)

        # JSON 변환 시도
        try:
            final_json = json.loads(structured_json)
        except json.JSONDecodeError:
            raise ValueError("GPT가 올바른 JSON을 반환하지 않았습니다.")

        # clothing_item이 카테고리 내에 없으면 None으로 설정
        clothing_item = final_json["features"].get("clothing_item")
        if clothing_item not in CATEGORIES:
            final_json["features"]["clothing_item"] = None

        # 최종 JSON에 시간 및 원본 문장 추가
        final_json["timestamp"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        final_json["speech_text"] = speech_text

        return json.dumps(final_json, ensure_ascii=False, indent=4)

    except Exception as e:
        return json.dumps({"error": "JSON 변환 실패", "message": str(e)}, ensure_ascii=False, indent=4)

# 실행 테스트
print(parse_speech_to_json(user_sentence))
