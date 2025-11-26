# utils/all_text_analyzer.py

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

    def __init__(self, model_name, max_tokens, temperature):
        self.hf_token: Optional[str] = get_hf_token()
        self.client: Optional[InferenceClient] = None
        self.model_name: str = model_name
        self.models_initialized: bool = False
        self.slides_per_block: int = 5
        self.max_tokens = max_tokens
        self.temperature = temperature

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
        После объединения пытаемся автоматом сопоставить найденные weaknesses/recommendations
        с номерами слайдов (если модель не указала их напрямую).
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
            raw = self._call_chat_model(prompt, max_tokens=self.max_tokens, temperature=self.temperature)
            parsed = self._try_parse_json(raw)
            if parsed:
                block_results.append(parsed)
            else:
                # fallback на блок
                block_results.append(self._fallback_summary(clean_text))

        # Объединяем результаты всех блоков
        combined = self._merge_block_results(block_results)

        # Попытка сопоставить элементы с номерами слайдов по содержимому,
        # только если модель сама не дала явные номера.
        try:
            combined = self._attach_slide_numbers_if_missing(combined, clean_text)
        except Exception as e:
            # не ломаем основной поток анализа, просто логируем
            print(f"[AllTextAnalyzer] slide mapping warning: {e}")

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
                # инициализация ключей если надо
                combined.setdefault(key, [])
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
                             "style","audience_level","quality_score","final_verdict"}
            if isinstance(data, dict) and expected_keys.issubset(set(data.keys())):
                data["strengths"] = [str(x).strip() for x in data.get("strengths", [])][:5]
                data["weaknesses"] = [str(x).strip() for x in data.get("weaknesses", [])][:20]
                data["recommendations"] = [str(x).strip() for x in data.get("recommendations", [])][:20]
                return data
        except json.JSONDecodeError:
            return None
        return None

    # ============================================================
    #  Postprocess: attach slide numbers if model didn't provide them
    # ============================================================
    def _attach_slide_numbers_if_missing(self, combined: Dict[str, Any], full_text: str) -> Dict[str, Any]:
        """
        Для каждого элемента в weaknesses/recommendations:
         - если элемент уже в формате {'slide':N,'text':...} — пропускаем
         - если строка содержит 'Слайд N' — парсим и превращаем в объект
         - иначе пытаемся найти подходящий слайд(ы) по содержанию (ngram/search)
        """
        slides = self._split_into_slides(full_text)  # list of dicts: {'num': int, 'text': str}
        combined = combined.copy()

        for key in ("weaknesses", "recommendations"):
            items = combined.get(key, [])
            processed = []
            for it in items:
                # если уже объект со slide — пропускаем
                if isinstance(it, dict) and ("slide" in it or "slides" in it):
                    processed.append(it)
                    continue
                s = str(it).strip()
                # 1) явный шаблон "Слайд N: ..."
                m = re.match(r"Слайд\s*(\d+)\s*[:\-–]\s*(.+)", s, flags=re.IGNORECASE)
                if m:
                    processed.append({"slide": int(m.group(1)), "text": m.group(2).strip()})
                    continue
                # 2) шаблон "Слайды N, M: ..." или "Слайды N–M: ..."
                m2 = re.match(r"Слайды\s*([\d,\s–\-]+)\s*[:\-–]\s*(.+)", s, flags=re.IGNORECASE)
                if m2:
                    nums = self._parse_slide_list(m2.group(1))
                    processed.append({"slides": nums, "text": m2.group(2).strip()})
                    continue
                # 3) пробуем сопоставить по содержанию
                mapped = self._map_text_to_slides_by_content(s, slides)
                if mapped:
                    # mapped — список номеров
                    if len(mapped) == 1:
                        processed.append({"slide": mapped[0], "text": s})
                    else:
                        processed.append({"slides": mapped, "text": s})
                else:
                    # ничего не найдено — оставляем строкой (фронтенд увидит, что нет номера)
                    processed.append(s)
            combined[key] = processed
        return combined

    def _split_into_slides(self, full_text: str) -> List[Dict[str, Any]]:
        """
        Разбивает full_text по маркерам '--- SLIDE N ---' в список словарей {'num':N,'text':...}
        """
        parts = re.split(r'--- SLIDE (\d+) ---', full_text)
        slides = []
        # parts: ['', '1', 'text1', '2', 'text2', ...] or maybe ['--- SLIDE 1 ---', ...] - handle robustly
        i = 0
        if len(parts) < 3:
            # fallback: разбить по страницам как строки
            lines = full_text.splitlines()
            slides.append({"num": 1, "text": "\n".join(lines)})
            return slides
        # iterate in pairs
        while i + 1 < len(parts):
            # parts pattern we get when splitting with capture: [prefix, num, text, num, text,...]
            # skip empty prefix possibly
            if parts[i].strip() == "" and i + 2 < len(parts):
                # parts[i+1] is number, parts[i+2] is text
                num = int(parts[i+1])
                text = parts[i+2]
                slides.append({"num": num, "text": text.lower()})
                i += 3
            else:
                # if starting differently
                try:
                    num = int(parts[i+1])
                    text = parts[i+2] if i+2 < len(parts) else ""
                    slides.append({"num": num, "text": text.lower()})
                    i += 3
                except Exception:
                    break
        if not slides:
            slides.append({"num": 1, "text": full_text.lower()})
        return slides

    def _parse_slide_list(self, s: str) -> List[int]:
        """
        Парсит фрагменты вида "1, 2, 4–6" -> [1,2,4,5,6]
        """
        parts = re.split(r'[,\s]+', s.strip())
        nums = []
        for p in parts:
            if not p:
                continue
            if "–" in p or "-" in p:
                bounds = re.split(r'[–\-]', p)
                try:
                    a = int(bounds[0])
                    b = int(bounds[1])
                    nums.extend(list(range(a, b+1)))
                except Exception:
                    continue
            else:
                try:
                    nums.append(int(p))
                except Exception:
                    continue
        return sorted(set(nums))

    def _map_text_to_slides_by_content(self, text: str, slides: List[Dict[str, Any]]) -> List[int]:
        """
        Попытка найти релевантные слайды по содержанию строки:
         - сначала ищем длинные ngram'ы (6..3 слов),
         - если не нашли — считаем пересечение значимых слов и выбираем топ-накопитель.
        Возвращаем список найденных номеров (может быть пуст).
        """
        text_l = text.lower()
        words = [w for w in re.findall(r'\w+', text_l) if len(w) > 2]
        if not words:
            return []

        # try n-grams (long to short)
        for n in range(min(6, len(words)), 2, -1):
            ngrams = [" ".join(words[i:i+n]) for i in range(0, len(words)-n+1)]
            for ng in ngrams:
                for s in slides:
                    if ng in s["text"]:
                        return [s["num"]]

        # если ngram не помог — простое совпадение слов
        scores = []
        slide_count = len(slides)
        for s in slides:
            slide_words = set(re.findall(r'\w+', s["text"]))
            # count intersection of meaningful words
            match_count = sum(1 for w in words if w in slide_words)
            scores.append((s["num"], match_count))

        # берём слайды с положительным совпадением, отсортированные по совпадениям
        positive = [num for num, cnt in sorted(scores, key=lambda x: -x[1]) if cnt > 0]
        # если слишком много совпадений — вернём топ-3
        if positive:
            return positive[:3]
        return []

    # ---- остальные fallback / вспомогательные методы -----------------------
    def _fallback_summary_from_text(self, raw_text: str, original_text: str) -> Dict[str, Any]:
        return {
            "main_topic": "Тема не определена",
            "goal": "Цель не определена",
            "summary": "Структурный анализ выполнен частично",
            "strengths": ["Стандартная структура слайдов"],
            "weaknesses": [{"slide": 1, "text": "Перегруженность текста"}],
            "recommendations": [{"slide": 1, "text": "Уменьшить количество текста"}],
            "structure_quality": "средняя",
            "clarity_score": 5,
            "style": "общий",
            "audience_level": "общая",
            "quality_score": 5,
            "final_verdict": "Fallback"
        }

    def _fallback_summary(self, text: str) -> Dict[str, Any]:
        return self._fallback_summary_from_text(text, text)

    def _normalize_full_text(self, text: str) -> str:
        return str(text or "").replace("\r\n", "\n").strip()

    def _clean_response(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text).strip()
