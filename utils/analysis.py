import os
import torch
from transformers import pipeline
from typing import Dict, Any, List
from PIL import Image
from core.config import get_hf_token
import asyncio

class PresentationAnalyzer:
    def __init__(self):
        self.hf_token = get_hf_token()
        self.text_analyzer = None
        self.models_initialized = False

    async def initialize_models(self):
        """Асинхронная инициализация моделей"""
        if self.models_initialized:
            return

        print("Начинаем инициализацию моделей...")

        try:
            self.text_analyzer = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-small",
                token=self.hf_token,
                device="cpu",
                dtype=torch.float32,
                pad_token_id=50256
            )

            self.models_initialized = True
            print("Модели успешно инициализированы")

        except Exception as e:
            print(f"Ошибка инициализации моделей: {e}")
            self.models_initialized = False

    def analyze_slide_content(self, text: str, image: Image = None) -> Dict[str, Any]:
        """Анализ содержания слайда"""
        if not self.models_initialized:
            return self._get_fallback_analysis(text, image)

        try:
            text_prompt = self._build_text_prompt(text)
            text_analysis = self._analyze_text(text_prompt)
            visual_analysis = self._analyze_visual(image) if image else {}

            return {
                "text_analysis": text_analysis,
                "visual_analysis": visual_analysis,
                "overall_score": self._calculate_overall_score(text_analysis, visual_analysis),
                "analysis_type": "ml_enhanced"
            }


        except Exception as e:
            print(f"ML анализ не удался, используем fallback: {e}")
            return self._get_fallback_analysis(text, image)

    def _get_fallback_analysis(self, text: str, image: Image = None) -> Dict[str, Any]:
        """Резервный анализ без ML"""
        word_count = len(text.split())

        return {
            "text_analysis": {
                "content_quality": "good",
                "key_messages": ["Базовый анализ текста"],
                "clarity_score": min(10, word_count // 50 + 5),
                "recommendations": ["Для детального анализа требуется ML модель"],
                "structure_analysis": "Базовый анализ завершен"
            },
            "visual_analysis": {
                "slide_dimensions": f"{image.width}x{image.height}" if image else "N/A",
                "text_amount": "умеренно",
                "layout_quality": "good",
                "visual_appeal_score": 6
            } if image else {},
            "overall_score": 6.5,
            "analysis_type": "fallback"
        }

    def _build_text_prompt(self, text: str) -> str:
        return f"""
        Кратко проанализируй текст слайда (максимум 3 предложения):

        {text[:500]}

        Основные моменты:
        """

    def _analyze_text(self, prompt: str) -> Dict[str, Any]:
        try:
            response = self.text_analyzer(
                prompt,
                max_new_tokens=100,
                num_return_sequences=1,
                temperature=0.3,
                do_sample=True,
                pad_token_id=50256
            )

            return self._parse_text_response(response[0]['generated_text'])

        except Exception as e:
            return {"error": f"Text analysis failed: {str(e)}"}

    def _analyze_visual(self, image: Image) -> Dict[str, Any]:
        try:
            width, height = image.size

            return {
                "slide_dimensions": f"{width}x{height}",
                "color_mode": image.mode,
                "text_amount": self._estimate_text_amount(image),
                "layout_quality": "good",
                "has_visual_elements": self._has_visual_elements(image),
                "visual_appeal_score": 7,
                "recommendations": ["Проверьте контрастность текста"]
            }
        except Exception as e:
            return {"error": f"Visual analysis failed: {str(e)}"}

    def _estimate_text_amount(self, image: Image) -> str:
        width, height = image.size
        total_pixels = width * height

        if total_pixels < 1000000:
            return "мало"
        elif total_pixels < 2000000:
            return "умеренно"
        else:
            return "много"

    def _has_visual_elements(self, image: Image) -> bool:
        width, height = image.size
        return width > 500 and height > 500

    def _parse_text_response(self, response: str) -> Dict[str, Any]:
        generated_text = response.replace(prompt, "").strip() if 'prompt' in locals() else response

        return {
            "content_quality": "good",
            "key_messages": [f"Анализ: {generated_text[:150]}"],
            "clarity_score": 7,
            "recommendations": ["Сократите текст", "Добавьте примеры"],
            "structure_analysis": "Логичная структура",
            "raw_response": generated_text[:200]  # для отладки
        }

    def _calculate_overall_score(self, text_analysis: Dict, visual_analysis: Dict) -> float:
        text_score = text_analysis.get('clarity_score', 5)
        visual_score = visual_analysis.get('visual_appeal_score', 5)
        return round((text_score + visual_score) / 2, 1)

    def generate_summary_report(self, slides_analysis: List[Dict]) -> Dict[str, Any]:
        if not self.models_initialized:
            return self._get_fallback_summary(slides_analysis)

        try:
            summary_prompt = f"Краткий итог по {len(slides_analysis)} слайдам:"

            response = self.text_analyzer(
                summary_prompt,
                max_new_tokens=150,
                pad_token_id=50256
            )

            return self._parse_text_response(response[0]['generated_text'])
        except Exception as e:
            return self._get_fallback_summary(slides_analysis)

    def _get_fallback_summary(self, slides_analysis: List[Dict]) -> Dict[str, Any]:
        total_slides = len(slides_analysis)
        avg_score = sum(
            slide.get('analysis', {}).get('overall_score', 6)
            for slide in slides_analysis
        ) / total_slides if total_slides > 0 else 6.5

        return {
            "total_score": round(avg_score, 1),
            "strengths": ["Хорошая структура", "Понятное изложение"],
            "improvements": ["Добавить визуальные элементы", "Увеличить детализацию"],
            "target_audience": "Общая аудитория",
            "analysis_type": "fallback"
        }


analyzer = PresentationAnalyzer()