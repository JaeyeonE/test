import io
import os
import json
from google.cloud import vision
from google.protobuf.json_format import MessageToDict


class VisionOCR:
    def __init__(self, credentials_path=None):
        """
        Google Vision API 클라이언트 초기화
        
        Args:
            credentials_path (str): 서비스 계정 JSON 파일 경로
        """
        if credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        
        self.client = vision.ImageAnnotatorClient()
    
    def extract_text_from_image(self, image_path):
        """
        이미지에서 텍스트 추출
        
        Args:
            image_path (str): OCR을 적용할 이미지 파일 경로
            
        Returns:
            dict: 추출된 텍스트와 전체 OCR 결과를 포함한 딕셔너리
        """
        try:
            # 이미지 로드
            with io.open(image_path, "rb") as image_file:
                content = image_file.read()
            
            image = vision.Image(content=content)
            
            # OCR 요청 (문서 내 텍스트 검출)
            response = self.client.document_text_detection(image=image)
            
            if response.error.message:
                raise Exception(f"Vision API 오류: {response.error.message}")
            
            # OCR 결과 추출
            if response.text_annotations:
                extracted_text = response.text_annotations[0].description
                response_dict = MessageToDict(response._pb)
                
                return {
                    "success": True,
                    "extracted_text": extracted_text,
                    "full_response": response_dict,
                    "error": None
                }
            else:
                return {
                    "success": False,
                    "extracted_text": None,
                    "full_response": None,
                    "error": "OCR 결과 없음"
                }
                
        except Exception as e:
            return {
                "success": False,
                "extracted_text": None,
                "full_response": None,
                "error": str(e)
            }
    
    def save_ocr_results(self, ocr_result, text_filename="ocr_result.txt", json_filename="ocr_result.json"):
        """
        OCR 결과를 파일로 저장
        
        Args:
            ocr_result (dict): extract_text_from_image 메서드의 반환값
            text_filename (str): 텍스트 파일명
            json_filename (str): JSON 파일명
        """
        if not ocr_result["success"]:
            print(f"❌ OCR 결과 저장 실패: {ocr_result['error']}")
            return False
        
        try:
            # 텍스트 파일 저장
            with open(text_filename, "w", encoding="utf-8") as text_file:
                text_file.write(ocr_result["extracted_text"])
            print(f"✅ OCR 텍스트 저장 완료: {text_filename}")
            
            # JSON 파일 저장
            with open(json_filename, "w", encoding="utf-8") as json_file:
                json.dump(ocr_result["full_response"], json_file, indent=4, ensure_ascii=False)
            print(f"✅ OCR 결과 JSON 저장 완료: {json_filename}")
            
            return True
            
        except Exception as e:
            print(f"❌ 파일 저장 오류: {str(e)}")
            return False