
import json
import io
import re
from typing import List, Dict, Any, Optional
from PIL import Image
from huggingface_hub import InferenceClient


from core.config import get_hf_token


class ImageAnalyzer:

    def __init__(self, model_name):
        self.hf_token: Optional[str] = get_hf_token()
        self.vlm_client: Optional[InferenceClient] = None
        self.llm_client: Optional[InferenceClient] = None

        self.caption_model = model_name
        self.reasoning_model = "IlyaGusev/saiga_llama3_8b"

        self.models_initialized = False

    async def initialize_models(self):
        if self.models_initialized:
            return
        try:
            self.vlm_client = InferenceClient(model=self.caption_model, token=self.hf_token)
            self.llm_client = InferenceClient(token=self.hf_token)
            self.models_initialized = True
            print(f"[ImageAnalyzer] InferenceClient ready (model {self.caption_model})")
        except Exception as e:
            print(f"[ImageAnalyzer] init error: {e}")
            self.models_initialized = False

    async def analyze_visual_presentation(self, slide_images: List[Image.Image]) -> Dict[str, Any]:

        if not self.models_initialized:
            return self._fallback()

        slide_results = []

        for idx, img in enumerate(slide_images, start=1):
            info = {"slide_number": idx}

            try:
                info["caption"] = self._caption(img)
            except:
                info["caption"] = ""

            stats = self._estimate_text_density(img)
            info.update(stats)

            if info["text_coverage"] > 0.35:
                info["slide_type"] = "text_heavy"
            elif info["text_coverage"] < 0.08:
                info["slide_type"] = "image_heavy"
            else:
                info["slide_type"] = "balanced"

            slide_results.append(info)

        prompt = self._build_global_prompt(slide_results)
        raw = self._call_llm(prompt)
        parsed = self._try_parse_json(raw)

        if parsed:
            return parsed

        return self._fallback()

    def _caption(self, img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        try:
            resp = self.vlm_client.image_to_text(buf)
            return resp.get("generated_text", "").strip()
        except:
            return ""

    def _estimate_text_density(self, img: Image.Image) -> Dict[str, float]:
        gray = img.convert("L")
        hist = gray.histogram()
        dark = sum(hist[:70])
        total = sum(hist)
        density = dark / total if total else 0
        coverage = min(1.0, density * 1.8)
        return {"text_density": round(density, 4), "text_coverage": round(coverage, 4)}



    def _build_global_prompt(self, slides: List[Dict[str, Any]]) -> str:
        example_json = {
            "strengths": ["Сильная композиция на слайде 1"],
            "weaknesses": ["Слайды 2, 5: слишком много текста"],
            "recommendations": ["Уменьшить текст на слайдах 2 и 5"],
            "design_style": "Профессиональный, современный",
            "quality_score": 80,
            "final_verdict": "Презентация в целом хороша, есть мелкие недочёты"
        }

        return (
            "Ты — эксперт по дизайну презентаций.\n"
            "Анализируй только визуальные характеристики слайдов (не текст и не смысл).\n"
            "Всегда возвращай ответ в формате JSON, ключи оставляй английскими, "
            "а текст внутри всех полей — строго на русском языке.\n\n"
            "Данные по слайдам:\n"
            f"{json.dumps(slides, ensure_ascii=False)}\n\n"
            "Пример правильного JSON-ответа:\n"
            f"{json.dumps(example_json, ensure_ascii=False)}\n\n"
            "Укажи в recommendations и visual_weaknesses конкретные номера слайдов с проблемами."
        )


    def _call_llm(self, prompt: str) -> str:
        try:
            resp = self.llm_client.chat_completion(
                model=self.reasoning_model,
                messages=[
                    {"role": "system", "content": "Ты — эксперт по визуальному анализу презентаций. Всегда отвечай на русском языке. Формат ответа — строго JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.0
            )
            if isinstance(resp, dict):
                ch = resp.get("choices") or resp.get("outputs")
                if ch:
                    msg = ch[0].get("message") or ch[0]
                    return msg.get("content") if isinstance(msg, dict) else str(msg)
            return str(resp)
        except Exception as e:
            print(f"[ImageAnalyzer] LLM error: {e}")
            return ""


    def _try_parse_json(self, text: str):
        if not text:
            return None
        cleaned = re.sub(r'```(?:json)?', '', text)
        cleaned = cleaned.replace("```", "")
        try:
            return json.loads(cleaned)
        except:
            return None

    def _fallback(self):
        return {
            "strengths": ["Невозможно выполнить анализ"],
            "weaknesses": ["Технический сбой"],
            "recommendations": ["Попробуйте позже"],
            "design_style": "неопределён",
            "quality_score": 5,
            "final_verdict": "Fallback"
        }

