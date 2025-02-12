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
    "Short Sl. Shirt", "Long Sl. Shirt", "Short Sl. Outw.", "Long Sl. Outw.",
    "Vest", "Sling", "Shorts", "Trousers", "Skirt", "Short Sl. Dress",
    "Long Sl. Dress", "Vest Dress", "Sling Dress"
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
user_sentence = "저 긴팔티 알려줘"
user_sentence = "날씨 알려줘"

############################################## STEP 1 ##############################################
def classify_speech_request(speech_text):
    """GPT를 활용해 발화가 의류 요청인지 판별"""
    prompt = f"""
    사용자가 AI 스피커에 요청한 문장이 제공됩니다. 
    이 문장이 TV 화면 속 의상에 대해 묻는 요청인지 판별해주세요.

    예시:
    - "저 남자가 입고 있는 검은색 셔츠 마음에 들어" -> 의류 요청 (Yes)
    - "오늘 날씨 어때?" -> 의류 요청 아님 (No)
    
    **답변 형식**:
    - "Yes" (의류 요청)
    - "No" (의류 요청 아님)
    
    **사용자 요청:** "{speech_text}"
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "당신은 문장이 의류 요청인지 아닌지 판별하는 도우미입니다."},
                      {"role": "user", "content": prompt}]
        )
        
        result = response.choices[0].message.content.strip()
        return result.lower() == "yes"
    except Exception as e:
        return False

############################################## STEP 2 ##############################################
def parse_speech_to_json(speech_text):
    """유저 발화 내용을 JSON 형태로 변환"""
    is_clothing_request = classify_speech_request(speech_text)
    
    if not is_clothing_request:
        return json.dumps({
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "speech_text": speech_text,
            "is_clothing_request": False,
            "message": "ERROR 유저 발화가 TV 화면의 의상 요청이 아닙니다."
        }, ensure_ascii=False, indent=4)
    
    prompt = f"""
    TV에 등장하는 의상에 대해 간단히 설명하는 문장을 제공할 것입니다. 
    이를 JSON 형식으로 변환해주세요. 
    지시 사항
    1. 의류 종류(`clothing_item`)는 아래 리스트에서만 선택 가능합니다. (단 반드시 하나의 CATEGORIES를 골라야만 합니다.):
    
    가능 항목:
    {CATEGORIES}
    
    2. 의류 색상(`color`)는 아래 리스트에서만 선택 가능합니다. (단 반드시 하나의 CATEGORIE를 골라야만 합니다.):
    
    가능 항목:
    {COLORS}

    JSON 구조는 다음과 같습니다:
    
    **JSON 예시**
    {{
        "features": {{
            "clothing_item": "CATEGORIES 항목 중 하나",
            "color": "COLORS 항목 중 하나",
            "location": "왼쪽, 오른쪽 등의 위치 (없으면 null)"
        }}
    }}
    
    **입력 문장:** "{speech_text}"
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "당신은 텍스트 문장을 JSON으로 변환하는 도우미입니다."},
                      {"role": "user", "content": prompt}]
        )
        
        structured_json = response.choices[0].message.content.strip()
        structured_json = re.sub(r"```json\n(.*?)\n```", r"\1", structured_json, flags=re.DOTALL)
        final_json = json.loads(structured_json)
        
        if final_json["features"].get("clothing_item") not in CATEGORIES:
            final_json["features"]["clothing_item"] = None
        if final_json["features"].get("color") not in COLORS:
            final_json["features"]["color"] = None
        
        final_json["timestamp"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        final_json["speech_text"] = speech_text
        final_json["is_clothing_request"] = True
        
        return json.dumps(final_json, ensure_ascii=False, indent=4)
    
    except Exception as e:
        return json.dumps({
            "error": "JSON 변환 실패",
            "message": str(e),
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "speech_text": speech_text,
            "is_clothing_request": is_clothing_request
        }, ensure_ascii=False, indent=4)

############################################## Output ##############################################
# 실행 테스트
print(parse_speech_to_json(user_sentence))