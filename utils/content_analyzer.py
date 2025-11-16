import json
import re
from typing import Dict, Any, Optional
from huggingface_hub import InferenceClient
from core.config import get_hf_token


class ContentAnalyzer:
    """
    Анализирует содержание всей презентации.
    Цель: выдавать рекомендации, ключевые моменты, слабые стороны и summary для студентов.
    Вход: текст всех слайдов с разделителями '--- SLIDE N ---'.
    """

    def __init__(self):
        self.hf_token: Optional[str] = get_hf_token()
        self.client: Optional[InferenceClient] = None
        self.model_name: str = "IlyaGusev/saiga_llama3_8b"
        self.models_initialized: bool = False

    async def initialize_models(self):
        if self.models_initialized:
            return
        try:
            self.client = InferenceClient(token=self.hf_token)
            self.models_initialized = True
            print(f"[ContentAnalyzer] InferenceClient ready (model {self.model_name})")
        except Exception as e:
            print(f"[ContentAnalyzer] init error: {e}")
            self.models_initialized = False

    def analyze_full_content(self, full_text: str) -> Dict[str, Any]:
        """
        Анализ содержания всей презентации. Возвращает словарь с ключевыми полями:
        - main_topic
        - summary
        - key_points
        - weaknesses
        - recommendations
        """
        clean_text = self._normalize_full_text(full_text)
        if not self.models_initialized or not self.client:
            return self._fallback_summary(clean_text)

        prompt = self._build_prompt_for_content_analysis(clean_text)
        raw = self._call_chat_model(prompt, max_tokens=800, temperature=0.0)

        parsed = self._try_parse_json(raw)
        if parsed:
            return parsed

        return self._fallback_summary_from_text(raw, clean_text)

    def _build_prompt_for_content_analysis(self, text: str) -> str:
        """
        Формируем промт для анализа содержания:
        - анализ ключевых идей
        - рекомендации для студентов
        - слабые стороны текста
        """
        instruction = (
            "Ты — преподаватель и эксперт по обучающим презентациям. "
            "Проанализируй текст всей презентации, разделённый слайдами '--- SLIDE N ---'.\n"
            "Верни строго JSON со следующей схемой:\n"
            "{\n"
            '  "main_topic": string,  # основная тема презентации\n'
            '  "summary": string,     # краткая выжимка содержания\n'
            '  "key_points": [string,...],  # ключевые моменты\n'
            '  "weaknesses": [string,...],  # недочёты и ошибки содержания, с указанием слайдов\n'
            '  "recommendations": [string,...] # советы, как улучшить содержание\n'
            "}\n"
            "Не добавляй markdown, code-blocks или лишние поля."
        )
        return instruction + "\n\n" + text

    def _call_chat_model(self, prompt: str, max_tokens: int = 800, temperature: float = 0.0) -> str:
        if not self.client:
            return ""
        try:
            response = self.client.chat_completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
            )
            text_out = ""
            if isinstance(response, dict):
                choices = response.get("choices") or response.get("outputs")
                if choices and isinstance(choices, list) and len(choices) > 0:
                    first = choices[0]
                    msg = first.get("message") or first
                    if isinstance(msg, dict):
                        text_out = msg.get("content") or msg.get("text") or ""
                    else:
                        text_out = str(first)
                else:
                    text_out = response.get("generated_text", "") or response.get("text", "") or ""
            else:
                text_out = str(response)
            return self._clean_response(text_out)
        except Exception as e:
            print(f"[ContentAnalyzer] LLM call error: {e}")
            return ""

    def _try_parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        t = re.sub(r'```(?:json)?\s*', '', text)
        t = re.sub(r'```', '', t)
        try:
            data = json.loads(t)
            expected = {"main_topic", "summary", "key_points", "weaknesses", "recommendations"}
            if isinstance(data, dict) and expected.issubset(set(data.keys())):
                # sanitize arrays -> strings
                data["key_points"] = [str(x).strip() for x in data.get("key_points", [])][:6]
                data["weaknesses"] = [str(x).strip() for x in data.get("weaknesses", [])][:6]
                data["recommendations"] = [str(x).strip() for x in data.get("recommendations", [])][:6]
                return data
        except json.JSONDecodeError:
            return None
        return None

    def _fallback_summary_from_text(self, raw_text: str, original_text: str) -> Dict[str, Any]:
        key_points = []
        weaknesses = []
        recommendations = []

        for line in re.split(r'[\n\r]+', raw_text):
            l = line.strip()
            if not l:
                continue
            low = l.lower()
            if "ключ" in low or "основн" in low:
                key_points.append(l)
            if "слаб" in low or "недостат" in low:
                weaknesses.append(l)
            if "рекоменд" in low or "совет" in low or "предлож" in low:
                recommendations.append(l)

        return {
            "main_topic": "Тема не определена",
            "summary": raw_text[:400] if raw_text else original_text[:400],
            "key_points": key_points[:5] or ["Ключевые моменты не определены"],
            "weaknesses": weaknesses[:5] or ["Недочёты не определены"],
            "recommendations": recommendations[:5] or ["Рекомендации не определены"]
        }

    def _fallback_summary(self, text: str) -> Dict[str, Any]:
        sentences = [s.strip() for s in re.split(r'[.!?]\s*', text) if s.strip()]
        return {
            "main_topic": sentences[0] if sentences else "Тема не определена",
            "summary": " ".join(sentences[:5]) or "Содержимое отсутствует",
            "key_points": sentences[:5] or ["Ключевые моменты не определены"],
            "weaknesses": ["Недочёты не определены"],
            "recommendations": ["Рекомендации не определены"]
        }

    def _normalize_full_text(self, text: str) -> str:
        t = str(text or "")
        t = re.sub(r'\r\n', '\n', t)
        return t.strip()

    def _clean_response(self, text: str) -> str:
        if not text:
            return ""
        t = text.strip()
        t = re.sub(r'[\x00-\x1f]+', ' ', t)
        return re.sub(r'\s+', ' ', t).strip()


content_analyzer = ContentAnalyzer()
