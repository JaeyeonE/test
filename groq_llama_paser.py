import os
import json
import requests
from typing import Dict, Optional


class GroqLLaMAParser:
    """
    Groq API를 활용한 LLaMA 모델 기반 텍스트 파서 클래스
    OCR로 추출된 명함 정보를 구조화하거나 커스텀 프롬프트로 텍스트를 분석할 수 있습니다.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Groq LLaMA API 클라이언트 초기화
        
        Args:
            api_key (str): Groq API 키. 없으면 환경변수에서 가져옴
        """
        # API 키를 매개변수로 받거나 환경변수에서 가져옴
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        
        # Groq API의 채팅 완료 엔드포인트 설정 (OpenAI 호환 형식)
        self.endpoint = "https://api.groq.com/openai/v1/chat/completions"
        
        # API 키가 없으면 예외 발생
        if not self.api_key:
            raise ValueError("GROQ_API_KEY가 설정되지 않았습니다. 환경변수나 매개변수로 제공해주세요.")
    
    def classify_business_card_info(self, text: str, model: str = "llama3-70b-8192") -> Dict:
        """
        명함에서 추출된 OCR 텍스트를 분석하여 구조화된 정보로 분류
        
        Args:
            text (str): OCR로 추출된 텍스트
            model (str): 사용할 Groq 모델명 (기본값: llama3-70b-8192)
            
        Returns:
            dict: 분류된 명함 정보 또는 오류 정보
                - success (bool): 성공 여부
                - raw_response (str): AI 모델의 원본 응답
                - error (str|None): 오류 메시지 (오류 발생 시)
        """
        # HTTP 요청 헤더 설정 (JSON 형식, Bearer 토큰 인증)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # API 요청 페이로드 구성
        payload = {
            "model": model,  # 사용할 LLaMA 모델 지정
            "messages": [
                {
                    "role": "system",
                    # 시스템 프롬프트: AI에게 명함 정보 추출 방법 지시
                    "content": """명함에서 추출된 OCR 텍스트를 분석하여 다음 항목들을 JSON 형식으로 추출해주세요:
                    - name: 한글 이름
                    - phone: 전화번호
                    - email: 이메일
                    - social_id: 소셜 미디어 ID (카카오톡 등)
                    - position: 직위/직책
                    - company: 회사/기관명 (있는 경우)
                    - address: 주소 (있는 경우)
                    - fax: 팩스 번호 (있는 경우)

                    출력은 JSON 형식으로만 제공해주세요. 정보가 없는 필드는 null로 표시하세요."""
                },
                {
                    "role": "user",
                    # 사용자 메시지: 실제 분석할 OCR 텍스트 전달
                    "content": f"다음은 명함에서 OCR로 추출한 텍스트입니다. 구조화된 정보로 분류해주세요:\n\n{text}"
                }
            ],
            "temperature": 0.3,    # 응답의 창의성 수준 (낮을수록 일관된 결과)
            "max_tokens": 1024     # 최대 응답 토큰 수 제한
        }
        
        try:
            # Groq API에 POST 요청 전송
            response = requests.post(self.endpoint, headers=headers, json=payload)
            
            # HTTP 에러 발생 시 예외 발생
            response.raise_for_status()
            
            # 응답을 JSON으로 파싱
            result = response.json()
            
            # 응답 구조 검증 및 결과 추출
            if "choices" in result and len(result["choices"]) > 0:
                # AI 모델의 실제 응답 텍스트 추출
                ai_response = result["choices"][0]["message"]["content"]
                return {
                    "success": True,
                    "raw_response": ai_response,  # AI의 원본 응답
                    "error": None
                }
            else:
                # 예상하지 못한 응답 형식일 경우
                return {
                    "success": False,
                    "raw_response": result,
                    "error": "API 응답에 예상된 형식이 없습니다."
                }
                
        except requests.exceptions.RequestException as e:
            # 네트워크 요청 관련 오류 처리
            return {
                "success": False,
                "raw_response": None,
                "error": f"API 요청 오류: {str(e)}"
            }
    
    def parse_custom_prompt(self, text: str, custom_prompt: str, model: str = "llama3-70b-8192") -> Dict:
        """
        커스텀 프롬프트를 사용하여 텍스트 분석
        사용자가 직접 정의한 분석 방식으로 텍스트를 처리할 수 있습니다.
        
        Args:
            text (str): 분석할 텍스트
            custom_prompt (str): 커스텀 시스템 프롬프트 (AI에게 주는 지시사항)
            model (str): 사용할 Groq 모델명 (기본값: llama3-70b-8192)
            
        Returns:
            dict: 분석 결과 또는 오류 정보
                - success (bool): 성공 여부
                - raw_response (str): AI 모델의 원본 응답
                - error (str|None): 오류 메시지 (오류 발생 시)
        """
        # HTTP 요청 헤더 설정 (JSON 형식, Bearer 토큰 인증)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # API 요청 페이로드 구성
        payload = {
            "model": model,  # 사용할 LLaMA 모델 지정
            "messages": [
                {
                    "role": "system",
                    # 사용자 정의 시스템 프롬프트 사용
                    "content": custom_prompt
                },
                {
                    "role": "user",
                    # 분석할 텍스트 전달
                    "content": text
                }
            ],
            "temperature": 0.3,    # 응답의 창의성 수준 (낮을수록 일관된 결과)
            "max_tokens": 1024     # 최대 응답 토큰 수 제한
        }
        
        try:
            # Groq API에 POST 요청 전송
            response = requests.post(self.endpoint, headers=headers, json=payload)
            
            # HTTP 에러 발생 시 예외 발생
            response.raise_for_status()
            
            # 응답을 JSON으로 파싱
            result = response.json()
            
            # 응답 구조 검증 및 결과 추출
            if "choices" in result and len(result["choices"]) > 0:
                # AI 모델의 실제 응답 텍스트 추출
                ai_response = result["choices"][0]["message"]["content"]
                return {
                    "success": True,
                    "raw_response": ai_response,  # AI의 원본 응답
                    "error": None
                }
            else:
                # 예상하지 못한 응답 형식일 경우
                return {
                    "success": False,
                    "raw_response": result,
                    "error": "API 응답에 예상된 형식이 없습니다."
                }
                
        except requests.exceptions.RequestException as e:
            # 네트워크 요청 관련 오류 처리
            return {
                "success": False,
                "raw_response": None,
                "error": f"API 요청 오류: {str(e)}"
            }