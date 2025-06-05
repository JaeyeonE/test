import json
import re
from typing import Dict, Any, Optional


class OutputCleaner:
    @staticmethod
    def extract_json_from_response(response: str) -> Dict[str, Any]:
        """
        LLM 응답에서 JSON 데이터 추출 및 파싱
        
        Args:
            response (str): LLM의 원시 응답 텍스트
            
        Returns:
            dict: 파싱된 JSON 데이터 또는 오류 정보
        """
        try:
            # JSON 코드 블록에서 추출
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].strip()
            else:
                json_str = response.strip()
            
            # JSON 파싱 시도
            parsed_data = json.loads(json_str)
            return {
                "success": True,
                "data": parsed_data,
                "error": None
            }
            
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "data": None,
                "error": f"JSON 파싱 오류: {str(e)}",
                "raw_response": response
            }
    
    @staticmethod
    def clean_business_card_data(raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        명함 데이터 정제 및 표준화
        
        Args:
            raw_data (dict): 원시 명함 데이터
            
        Returns:
            dict: 정제된 명함 데이터
        """
        cleaned_data = {}
        
        # 표준 필드 매핑
        field_mapping = {
            "name": ["name", "이름", "성명"],
            "phone": ["phone", "전화번호", "휴대폰", "연락처"],
            "email": ["email", "이메일", "메일"],
            "social_id": ["social_id", "카카오톡", "카톡", "sns"],
            "position": ["position", "직위", "직책", "역할"],
            "company": ["company", "회사", "기관", "업체"],
            "address": ["address", "주소", "소재지"],
            "fax": ["fax", "팩스", "팩시밀리"]
        }
        
        # 각 표준 필드에 대해 데이터 매핑
        for standard_field, possible_keys in field_mapping.items():
            value = None
            
            # 가능한 키들을 순서대로 확인
            for key in possible_keys:
                if key in raw_data and raw_data[key] is not None:
                    value = raw_data[key]
                    break
            
            # 값 정제
            if value:
                cleaned_data[standard_field] = OutputCleaner._clean_field_value(value, standard_field)
            else:
                cleaned_data[standard_field] = None
        
        return cleaned_data
    
    @staticmethod
    def _clean_field_value(value: Any, field_type: str) -> Optional[str]:
        """
        개별 필드 값 정제
        
        Args:
            value: 원시 값
            field_type (str): 필드 타입
            
        Returns:
            str or None: 정제된 값
        """
        if value is None or value == "":
            return None
        
        # 문자열로 변환
        str_value = str(value).strip()
        
        if not str_value or str_value.lower() in ['null', 'none', 'n/a', '-']:
            return None
        
        # 필드 타입별 특별 처리
        if field_type == "phone":
            return OutputCleaner._clean_phone_number(str_value)
        elif field_type == "email":
            return OutputCleaner._clean_email(str_value)
        else:
            return str_value
    
    @staticmethod
    def _clean_phone_number(phone: str) -> Optional[str]:
        """
        전화번호 정제
        
        Args:
            phone (str): 원시 전화번호
            
        Returns:
            str or None: 정제된 전화번호
        """
        # 숫자와 하이픈만 남기기
        cleaned = re.sub(r'[^\d\-+()]', '', phone)
        
        # 빈 문자열 체크
        if not cleaned:
            return None
        
        return cleaned
    
    @staticmethod
    def _clean_email(email: str) -> Optional[str]:
        """
        이메일 주소 정제
        
        Args:
            email (str): 원시 이메일
            
        Returns:
            str or None: 정제된 이메일
        """
        # 기본적인 이메일 형식 체크
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        if re.match(email_pattern, email.strip()):
            return email.strip().lower()
        else:
            return None
    
    @staticmethod
    def create_final_result(cleaned_data: Dict[str, Any], card_id: int = 1, 
                          image_path: str = None, profile_image_url: str = None) -> Dict[str, Any]:
        """
        최종 결과 데이터 구조 생성
        
        Args:
            cleaned_data (dict): 정제된 명함 데이터
            card_id (int): 카드 ID
            image_path (str): 명함 이미지 경로
            profile_image_url (str): 프로필 이미지 URL
            
        Returns:
            dict: 최종 명함 데이터 구조
        """
        return {
            "cardId": card_id,
            "name": cleaned_data.get("name"),
            "phone": cleaned_data.get("phone"),
            "email": cleaned_data.get("email"),
            "profile_image_url": profile_image_url,
            "card_img_url": image_path,
            "address": cleaned_data.get("address"),
            "fax": cleaned_data.get("fax"),
            "position": cleaned_data.get("position"),
            "company": cleaned_data.get("company"),
            "social_id": cleaned_data.get("social_id")
        }
    
    @staticmethod
    def save_result(data: Dict[str, Any], filename: str = "card_classified_data.json") -> bool:
        """
        결과 데이터를 JSON 파일로 저장
        
        Args:
            data (dict): 저장할 데이터
            filename (str): 저장할 파일명
            
        Returns:
            bool: 저장 성공 여부
        """
        try:
            with open(filename, "w", encoding="utf-8") as result_file:
                json.dump(data, result_file, indent=4, ensure_ascii=False)
            print(f"✅ 명함 분류 결과 저장 완료: {filename}")
            return True
        except Exception as e:
            print(f"❌ 파일 저장 오류: {str(e)}")
            return False