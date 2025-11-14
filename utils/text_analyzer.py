import os
import torch
from transformers import pipeline, AutoTokenizer, T5ForConditionalGeneration
from typing import Dict, Any, List
from core.config import get_hf_token
import asyncio
import re


class TextAnalyzer:
    def __init__(self):
        self.hf_token = get_hf_token()
        self.model = None
        self.tokenizer = None
        self.models_initialized = False

    async def initialize_models(self):
        if self.models_initialized:
            return

        print("Инициализация модели...")

        try:
            model_name = "sberbank-ai/rugpt3small_based_on_gpt2"

            self.pipeline = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=model_name,
                token=self.hf_token,
                device="cpu",
                torch_dtype=torch.float32,
            )

            self.models_initialized = True
            print("Модель инициализирована")

        except Exception as e:
            print(f"Ошибка инициализации модели: {e}")
            # Fallback на очень простую модель
            try:
                self.pipeline = pipeline(
                    "text-generation",
                    model="distilgpt2",
                    device="cpu"
                )
                self.models_initialized = True
                print("Загружена резервная модель")
            except Exception as fallback_error:
                print(f"Не удалось загрузить даже резервную модель: {fallback_error}")
                self.models_initialized = False

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Анализ текста слайда"""
        if not self.models_initialized:
            return self._get_fallback_analysis(text)

        try:
            clean_text = self._clean_text(text)

            analysis = self._get_meaningful_analysis(clean_text)
            recommendations = self._get_meaningful_recommendations(clean_text)
            problems = self._get_meaningful_problems(clean_text)

            return {
                "main_topic": self._extract_main_topic(clean_text),
                "key_points": self._extract_key_points(clean_text),
                "clarity_score": self._calculate_clarity_score(clean_text),
                "structure_quality": self._assess_structure(clean_text),
                "specific_recommendations": recommendations,
                "problems_detected": problems,
                "llm_analysis": analysis,
                "analysis_type": "russian_llm"
            }

        except Exception as e:
            print(f"Ошибка анализа текста: {e}")
            return self._get_fallback_analysis(text)

    def _clean_text(self, text: str) -> str:
        """Очистка текста"""
        # Убираем лишние переносы и пробелы
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()[:400]  # Ограничиваем длину

    def _get_meaningful_analysis(self, text: str) -> str:
        """Получение осмысленного анализа"""
        prompt = f"""
        Текст слайда презентации: "{text}"

        Кратко проанализируй этот текст. О чем он? Какая основная идея?
        Анализ:
        """
        return self._safe_llm_call(prompt, 60)

    def _get_meaningful_recommendations(self, text: str) -> List[str]:
        """Получение осмысленных рекомендаций"""
        prompt = f"""
        Текст слайда: "{text}"

        Дай 2 практические рекомендации по улучшению этого слайда. Будь конкретен.
        Рекомендации:
        1.
        """
        response = self._safe_llm_call(prompt, 80)
        return self._parse_meaningful_list(response, "Улучшите структуру изложения")

    def _get_meaningful_problems(self, text: str) -> List[str]:
        """Получение осмысленных проблем"""
        prompt = f"""
        Текст слайда: "{text}"

        Найди 2 основные проблемы в этом тексте для презентации.
        Проблемы:
        1.
        """
        response = self._safe_llm_call(prompt, 80)
        return self._parse_meaningful_list(response, "Проверьте ясность изложения")

    def _safe_llm_call(self, prompt: str, max_tokens: int) -> str:
        """Безопасный вызов LLM"""
        try:
            prompt = prompt.strip()[:600]

            response = self.pipeline(
                prompt,
                max_new_tokens=max_tokens,
                num_return_sequences=1,
                temperature=0.3,
                do_sample=True,
                pad_token_id=50256,
                truncation=True,
                repetition_penalty=1.3
            )

            if response and len(response) > 0:
                generated_text = response[0]['generated_text']

                # Извлекаем только ответ
                if prompt in generated_text:
                    response_text = generated_text.replace(prompt, "").strip()
                else:
                    response_text = generated_text.strip()

                # Очищаем ответ
                return self._clean_response(response_text)

            return ""

        except Exception as e:
            print(f"Ошибка LLM: {e}")
            return ""

    def _clean_response(self, text: str) -> str:
        """Очистка ответа"""
        text = re.sub(r'[^\w\sа-яА-ЯёЁ.,!?;:()-]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _parse_meaningful_list(self, response: str, default: str) -> List[str]:
        """Парсинг списка"""
        if not response:
            return [default]

        lines = []
        for part in re.split(r'[\n\.]', response):
            part = part.strip()
            clean_part = re.sub(r'^[\d\-•*]\.?\s*', '', part)
            if clean_part and len(clean_part) > 15 and len(clean_part) < 100:
                lines.append(clean_part)

        return lines[:2] if lines else [default]

    def _extract_main_topic(self, text: str) -> str:
        """Извлечение темы"""
        sentences = re.split(r'[.!?]+', text)
        if sentences:
            first_sentence = sentences[0].strip()
            if len(first_sentence) > 100:
                return first_sentence[:100] + "..."
            return first_sentence
        return "Тема не определена"

    def _extract_key_points(self, text: str) -> List[str]:
        """Извлечение ключевых пунктов"""
        sentences = re.split(r'[.!?]+', text)
        clean_sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 15]
        return clean_sentences[:3] if clean_sentences else ["Информация представлена в тексте"]

    def _calculate_clarity_score(self, text: str) -> int:
        """Оценка ясности"""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        clean_sentences = [s for s in sentences if s.strip()]

        if not clean_sentences:
            return 3

        avg_length = len(words) / len(clean_sentences)

        if 10 <= avg_length <= 25:
            return 8
        elif 5 <= avg_length < 10 or 25 < avg_length <= 35:
            return 6
        else:
            return 4

    def _assess_structure(self, text: str) -> str:
        """Оценка структуры"""
        sentences = re.split(r'[.!?]+', text)
        clean_sentences = [s for s in sentences if s.strip()]

        if len(clean_sentences) >= 3:
            return "хорошая"
        elif len(clean_sentences) >= 2:
            return "базовая"
        else:
            return "минимальная"

    def _get_fallback_analysis(self, text: str) -> Dict[str, Any]:
        """Резервный анализ"""
        clean_text = self._clean_text(text)
        return {
            "main_topic": self._extract_main_topic(clean_text),
            "key_points": self._extract_key_points(clean_text),
            "clarity_score": self._calculate_clarity_score(clean_text),
            "structure_quality": self._assess_structure(clean_text),
            "specific_recommendations": ["Для детального анализа требуется русскоязычная модель"],
            "problems_detected": ["Используется базовая обработка текста"],
            "llm_analysis": "Модель не загружена",
            "analysis_type": "fallback"
        }


text_analyzer = TextAnalyzer()