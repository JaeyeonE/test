#!/usr/bin/env python3
"""
명함 OCR 및 정보 추출 시스템
Google Vision API와 Groq LLaMA API를 사용하여 명함 이미지에서 정보를 추출하고 구조화합니다.
"""

import os
import json
import argparse
from typing import Dict, Any, Optional

from vision import VisionOCR
from groq_llama_parser import GroqLLaMAParser
from output_cleaner import OutputCleaner


class BusinessCardProcessor:
    def __init__(self, vision_credentials_path: Optional[str] = None, groq_api_key: Optional[str] = None):
        """
        명함 처리기 초기화
        
        Args:
            vision_credentials_path (str): Google Vision API 인증 파일 경로
            groq_api_key (str): Groq API 키
        """
        try:
            self.vision_ocr = VisionOCR(vision_credentials_path)
            self.groq_parser = GroqLLaMAParser(groq_api_key)
            self.output_cleaner = OutputCleaner()
            print("✅ 모든 API 클라이언트 초기화 완료")
        except Exception as e:
            print(f"❌ 초기화 오류: {str(e)}")
            raise
    
    def process_business_card(self, image_path: str, card_id: int = 1, 
                            save_intermediate: bool = True) -> Dict[str, Any]:
        """
        명함 이미지를 처리하여 구조화된 정보 추출
        
        Args:
            image_path (str): 명함 이미지 파일 경로
            card_id (int): 카드 ID
            save_intermediate (bool): 중간 결과 파일 저장 여부
            
        Returns:
            dict: 처리 결과 (성공/실패 정보 포함)
        """
        print(f"\n🔹 명함 처리 시작: {image_path}")
        
        # 1. OCR로 텍스트 추출
        print("1️⃣ OCR 텍스트 추출 중...")
        ocr_result = self.vision_ocr.extract_text_from_image(image_path)
        
        if not ocr_result["success"]:
            return {
                "success": False,
                "error": f"OCR 실패: {ocr_result['error']}",
                "step": "OCR"
            }
        
        extracted_text = ocr_result["extracted_text"]
        print(f"✅ OCR 완료. 추출된 텍스트 길이: {len(extracted_text)} 문자")
        
        if save_intermediate:
            self.vision_ocr.save_ocr_results(ocr_result)
        
        # 2. LLaMA로 정보 분류
        print("2️⃣ LLaMA API로 정보 분류 중...")
        classification_result = self.groq_parser.classify_business_card_info(extracted_text)
        
        if not classification_result["success"]:
            return {
                "success": False,
                "error": f"분류 실패: {classification_result['error']}",
                "step": "Classification"
            }
        
        print("✅ 분류 완료")
        
        # 3. JSON 파싱 및 정제
        print("3️⃣ 결과 정제 중...")
        json_result = self.output_cleaner.extract_json_from_response(classification_result["raw_response"])
        
        if not json_result["success"]:
            return {
                "success": False,
                "error": f"JSON 파싱 실패: {json_result['error']}",
                "step": "JSON Parsing",
                "raw_response": classification_result["raw_response"]
            }
        
        # 4. 데이터 정제 및 최종 결과 생성
        cleaned_data = self.output_cleaner.clean_business_card_data(json_result["data"])
        final_result = self.output_cleaner.create_final_result(
            cleaned_data, 
            card_id=card_id, 
            image_path=image_path
        )
        
        print("✅ 정제 완료")
        
        # 5. 결과 저장
        if save_intermediate:
            self.output_cleaner.save_result(final_result)
        
        return {
            "success": True,
            "data": final_result,
            "extracted_text": extracted_text,
            "error": None
        }
    
    def process_multiple_cards(self, image_paths: list, start_card_id: int = 1) -> Dict[str, Any]:
        """
        여러 명함 이미지를 일괄 처리
        
        Args:
            image_paths (list): 명함 이미지 파일 경로 리스트
            start_card_id (int): 시작 카드 ID
            
        Returns:
            dict: 일괄 처리 결과
        """
        results = []
        success_count = 0
        
        for i, image_path in enumerate(image_paths):
            card_id = start_card_id + i
            result = self.process_business_card(image_path, card_id, save_intermediate=False)
            
            results.append({
                "image_path": image_path,
                "card_id": card_id,
                "result": result
            })
            
            if result["success"]:
                success_count += 1
            else:
                print(f"❌ 실패: {image_path} - {result['error']}")
        
        # 일괄 결과 저장
        batch_result = {
            "total_processed": len(image_paths),
            "success_count": success_count,
            "failure_count": len(image_paths) - success_count,
            "results": results
        }
        
        with open("batch_processing_result.json", "w", encoding="utf-8") as f:
            json.dump(batch_result, f, indent=4, ensure_ascii=False)
        
        print(f"\n📊 일괄 처리 완료: {success_count}/{len(image_paths)} 성공")
        return batch_result


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="명함 OCR 및 정보 추출")
    parser.add_argument("image_path", help="명함 이미지 파일 경로")
    parser.add_argument("--card-id", type=int, default=1, help="카드 ID (기본값: 1)")
    parser.add_argument("--vision-credentials", help="Google Vision API 인증 파일 경로")
    parser.add_argument("--groq-api-key", help="Groq API 키")
    parser.add_argument("--no-save", action="store_true", help="중간 결과 파일 저장 안함")
    
    args = parser.parse_args()
    
    try:
        # 환경변수 로드 (.env 파일이 있다면)
        if os.path.exists(".env"):
            from dotenv import load_dotenv
            load_dotenv()
        
        # 프로세서 초기화
        processor = BusinessCardProcessor(
            vision_credentials_path=args.vision_credentials,
            groq_api_key=args.groq_api_key
        )
        
        # 명함 처리
        result = processor.process_business_card(
            image_path=args.image_path,
            card_id=args.card_id,
            save_intermediate=not args.no_save
        )
        
        # 결과 출력
        if result["success"]:
            print("\n🔹 최종 결과:")
            print(json.dumps(result["data"], indent=2, ensure_ascii=False))
        else:
            print(f"\n❌ 처리 실패: {result['error']}")
            return 1
            
    except Exception as e:
        print(f"❌ 실행 오류: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())