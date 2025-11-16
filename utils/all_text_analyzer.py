import json
import re
from typing import Dict, Any, List, Optional
from huggingface_hub import InferenceClient
from core.config import get_hf_token


class AllTextAnalyzer:
    """
    Анализирует структуру презентации: плотность текста, читаемость, заголовки.
    Поддерживает большие презентации за счет разбивки на блоки слайдов.
    """

    def __init__(self):
        self.hf_token: Optional[str] = get_hf_token()
        self.client: Optional[InferenceClient] = None
        self.model_name: str = "IlyaGusev/saiga_llama3_8b"
        self.models_initialized: bool = False
        self.slides_per_block: int = 5

    async def initialize_models(self) -> None:
        if self.models_initialized:
            return
        try:
            self.client = InferenceClient(token=self.hf_token)
            self.models_initialized = True
            print(f"[AllTextAnalyzer] InferenceClient ready (model {self.model_name})")
        except Exception as e:
            print(f"[AllTextAnalyzer] init error: {e}")
            self.models_initialized = False

    def analyze_full_text(self, full_text: str) -> Dict[str, Any]:
        """
        Анализ всей презентации.
        Разбиваем текст на блоки, генерируем JSON для каждого блока, потом объединяем.
        """
        clean_text = self._normalize_full_text(full_text)
        if not self.models_initialized or not self.client:
            return self._fallback_summary(clean_text)

        slide_texts = re.split(r'(--- SLIDE \d+ ---)', clean_text)
        blocks = self._make_blocks(slide_texts, self.slides_per_block)

        # Генерируем JSON для каждого блока
        block_results = []
        for block_text in blocks:
            prompt = self._build_prompt_for_structural_analysis(block_text)
            raw = self._call_chat_model(prompt, max_tokens=2000, temperature=0.0)
            parsed = self._try_parse_json(raw)
            if parsed:
                block_results.append(parsed)
            else:
                # fallback на блок
                block_results.append(self._fallback_summary(clean_text))

        # Объединяем результаты всех блоков
        combined = self._merge_block_results(block_results)
        return combined

    # ---- блокировка слайдов -------------------------------------------------
    def _make_blocks(self, slides: List[str], block_size: int) -> List[str]:
        """
        slides: ['--- SLIDE 1 ---', 'text1', '--- SLIDE 2 ---', 'text2', ...]
        Возвращает список текстов блоков по block_size слайдов.
        """
        blocks = []
        current_block = []
        count = 0
        for i in range(0, len(slides), 2):
            if i + 1 >= len(slides):
                break
            header = slides[i]
            text = slides[i + 1]
            current_block.append(f"{header}\n{text}")
            count += 1
            if count >= block_size:
                blocks.append("\n\n".join(current_block))
                current_block = []
                count = 0
        if current_block:
            blocks.append("\n\n".join(current_block))
        return blocks

    def _build_prompt_for_structural_analysis(self, text: str) -> str:
        instruction = (
            "Ты — эксперт по презентациям. Проанализируй структуру презентации (только текст и заголовки). "
            "Текст содержит все слайды, разделённые '--- SLIDE N ---'.\n\n"
            "Верни строго JSON с полями:\n"
            "- strengths: сильные стороны структуры презентации\n"
            "- weaknesses: слабые места (укажи номера слайдов с проблемами, например 'Слайд 2: ...')\n"
            "- recommendations: рекомендации по улучшению структуры с номерами слайдов\n"
            "Остальные поля: main_topic, goal, summary, structure_quality, clarity_score, style, "
            "audience_level, overall_quality_score, final_verdict.\n"
            "Не анализируй содержание текста, не добавляй markdown или code-blocks.\n"
        )
        return instruction + "\n\n" + text

    def _call_chat_model(self, user_prompt: str, max_tokens: int = 2000, temperature: float = 0.0) -> str:
        if not self.client:
            return ""
        try:
            response = self.client.chat_completion(
                model=self.model_name,
                messages=[{"role": "user", "content": user_prompt}],
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
                text_out = str(response)
            return self._clean_response(text_out)
        except Exception as e:
            print(f"[AllTextAnalyzer] LLM call error: {e}")
            return ""

    # ---- объединение результатов блоков ---------------------------------
    def _merge_block_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        combined = results[0] if results else {}
        if len(results) == 1:
            return combined

        for r in results[1:]:
            for key in ["strengths", "weaknesses", "recommendations"]:
                combined[key].extend(r.get(key, []))
            for key in ["clarity_score", "overall_quality_score"]:
                combined[key] = int(round((combined.get(key, 0) + r.get(key, 0)) / 2))
        return combined

    def _try_parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        t = re.sub(r'```(?:json)?\s*', '', text)
        t = re.sub(r'```', '', t)
        try:
            data = json.loads(t)
            expected_keys = {"main_topic","goal","summary","strengths","weaknesses",
                             "recommendations","structure_quality","clarity_score",
                             "style","audience_level","overall_quality_score","final_verdict"}
            if isinstance(data, dict) and expected_keys.issubset(set(data.keys())):
                data["strengths"] = [str(x).strip() for x in data.get("strengths", [])][:5]
                data["weaknesses"] = [str(x).strip() for x in data.get("weaknesses", [])][:5]
                data["recommendations"] = [str(x).strip() for x in data.get("recommendations", [])][:6]
                return data
        except json.JSONDecodeError:
            return None
        return None

    def _fallback_summary_from_text(self, raw_text: str, original_text: str) -> Dict[str, Any]:
        return {
            "main_topic": "Тема не определена",
            "goal": "Цель не определена",
            "summary": "Структурный анализ выполнен частично",
            "strengths": ["Стандартная структура слайдов"],
            "weaknesses": ["Слайды 1-2: перегруженность текста", "Слайды 3-4: заголовки неявные"],
            "recommendations": [
                "Слайды 1-2: уменьшить количество текста",
                "Слайды 3-4: добавить явные заголовки и разделители"
            ],
            "structure_quality": "средняя",
            "clarity_score": 5,
            "style": "общий",
            "audience_level": "общая",
            "overall_quality_score": 5,
            "final_verdict": "Режим fallback"
        }

    def _fallback_summary(self, text: str) -> Dict[str, Any]:
        return self._fallback_summary_from_text(text, text)

    def _normalize_full_text(self, text: str) -> str:
        return str(text or "").replace("\r\n", "\n").strip()

    def _clean_response(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text).strip()


all_text_analyzer = AllTextAnalyzer()
