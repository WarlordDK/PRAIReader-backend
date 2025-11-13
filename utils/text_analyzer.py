import os
import torch
from transformers import pipeline
from typing import Dict, Any, List
from core.config import get_hf_token
import asyncio
import re


class TextAnalyzer:
    def __init__(self):
        self.hf_token = get_hf_token()
        self.model = None
        self.models_initialized = False

    async def initialize_models(self):
        """Инициализация текстовой модели"""
        if self.models_initialized:
            return

        print("🔄 Инициализация текстовой модели...")

        try:
            # Используем маленькую но стабильную модель
            self.model = pipeline(
                "text-generation",
                model="distilgpt2",  # Надежная и быстрая модель
                device="cpu",
                torch_dtype=torch.float32,
            )

            self.models_initialized = True
            print("✅ Текстовая модель инициализирована")

        except Exception as e:
            print(f"❌ Ошибка инициализации текстовой модели: {e}")
            self.models_initialized = False

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Анализ текста слайда"""
        if not self.models_initialized:
            return self._get_fallback_analysis(text)

        try:
            # Ограничиваем текст для избежания ошибок
            short_text = text[:300]

            # Получаем ответы от LLM
            analysis = self._get_llm_analysis(short_text)
            recommendations = self._get_llm_recommendations(short_text)
            problems = self._get_llm_problems(short_text)

            return {
                "main_topic": self._extract_main_topic(text),
                "key_points": self._extract_key_points(text),
                "clarity_score": self._calculate_clarity_score(text),
                "structure_quality": self._assess_structure(text),
                "specific_recommendations": recommendations,
                "problems_detected": problems,
                "llm_analysis": analysis,
                "analysis_type": "llm_enhanced"
            }

        except Exception as e:
            print(f"Ошибка анализа текста: {e}")
            return self._get_fallback_analysis(text)

    def _get_llm_analysis(self, text: str) -> str:
        """Получение анализа от LLM"""
        prompt = f"Проанализируй текст слайда: '{text}'. Основные идеи:"
        return self._safe_llm_call(prompt, 80)

    def _get_llm_recommendations(self, text: str) -> List[str]:
        """Получение рекомендаций от LLM"""
        prompt = f"Дай рекомендации по тексту слайда: '{text}'. Советы:"
        response = self._safe_llm_call(prompt, 60)
        return self._parse_list_response(response, "Улучшите ясность изложения")

    def _get_llm_problems(self, text: str) -> List[str]:
        """Получение проблем от LLM"""
        prompt = f"Какие проблемы в тексте слайда: '{text}'? Недостатки:"
        response = self._safe_llm_call(prompt, 60)
        return self._parse_list_response(response, "Проблемы не выявлены")

    def _safe_llm_call(self, prompt: str, max_tokens: int) -> str:
        """Безопасный вызов LLM"""
        try:
            # Ограничиваем длину промта
            if len(prompt) > 500:
                prompt = prompt[:500]

            response = self.model(
                prompt,
                max_new_tokens=max_tokens,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=50256,
                truncation=True
            )

            if response and len(response) > 0:
                generated_text = response[0]['generated_text']
                # Убираем промпт из ответа
                if prompt in generated_text:
                    return generated_text.replace(prompt, "").strip()
                return generated_text[:150].strip()  # Ограничиваем длину ответа
            return ""

        except Exception as e:
            print(f"Ошибка LLM call: {e}")
            return ""

    def _parse_list_response(self, response: str, default: str) -> List[str]:
        """Парсинг ответа в список"""
        if not response:
            return [default]

        # Разбиваем на пункты
        lines = [line.strip() for line in response.split('.') if line.strip()]
        items = []

        for line in lines:
            clean_line = re.sub(r'^[\d\-•*]\s*', '', line).strip()
            if clean_line and len(clean_line) > 10 and len(clean_line) < 100:
                items.append(clean_line)

        return items[:2] if items else [default]

    def _extract_main_topic(self, text: str) -> str:
        """Извлечение основной темы"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if sentences:
            first_sentence = sentences[0]
            words = first_sentence.split()[:5]
            return ' '.join(words) + ('...' if len(first_sentence) > len(' '.join(words)) else '')
        return "Тема не определена"

    def _extract_key_points(self, text: str) -> List[str]:
        """Извлечение ключевых пунктов"""
        sentences = [s.strip() for s in text.split('.') if s.strip() and len(s.strip()) > 8]
        return sentences[:2] if sentences else ["Информация представлена в тексте"]

    def _calculate_clarity_score(self, text: str) -> int:
        """Оценка ясности"""
        words = text.split()
        sentences = [s for s in text.split('.') if s.strip()]

        if not sentences:
            return 3

        avg_length = len(words) / len(sentences)

        if 10 <= avg_length <= 25:
            return 8
        elif 5 <= avg_length < 10 or 25 < avg_length <= 35:
            return 6
        else:
            return 4

    def _assess_structure(self, text: str) -> str:
        """Оценка структуры"""
        sentences = [s for s in text.split('.') if s.strip()]
        if len(sentences) >= 3:
            return "хорошая"
        elif len(sentences) >= 2:
            return "базовая"
        else:
            return "минимальная"

    def _get_fallback_analysis(self, text: str) -> Dict[str, Any]:
        """Резервный анализ"""
        return {
            "main_topic": self._extract_main_topic(text),
            "key_points": self._extract_key_points(text),
            "clarity_score": self._calculate_clarity_score(text),
            "structure_quality": self._assess_structure(text),
            "specific_recommendations": ["Для детального анализа загрузите ML модель"],
            "problems_detected": ["Анализ выполнен в базовом режиме"],
            "llm_analysis": "Модель не загружена",
            "analysis_type": "fallback"
        }


# Глобальный экземпляр
text_analyzer = TextAnalyzer()