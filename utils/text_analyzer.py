# # utils/text_analyzer.py
# import json
# import re
# from typing import Dict, Any, List, Optional
#
# from huggingface_hub import InferenceClient
# from core.config import get_hf_token
#
#
# class TextAnalyzer:
#     """
#     Переработанный per-slide TextAnalyzer.
#     - Убирает specific_recommendations (перенесли общие рекомендации в all_text_analyzer)
#     - Запрашивает у LLM краткий анализ и проблемы (строками, без JSON/code-block)
#     - Возвращает словарь с полями, совместимыми с router.py
#     """
#
#     def __init__(self):
#         self.hf_token: Optional[str] = get_hf_token()
#         self.client: Optional[InferenceClient] = None
#         self.model_name: str = "IlyaGusev/saiga_llama3_8b"
#         self.models_initialized: bool = False
#
#     async def initialize_models(self) -> None:
#         if self.models_initialized:
#             return
#         try:
#             self.client = InferenceClient(token=self.hf_token)
#             self.models_initialized = True
#             print(f"[TextAnalyzer] InferenceClient ready (model {self.model_name})")
#         except Exception as e:
#             print(f"[TextAnalyzer] Failed to initialize InferenceClient: {e}")
#             self.models_initialized = False
#
#     def analyze_text(self, text: str) -> Dict[str, Any]:
#         clean = self._clean_text(text)
#
#         if not self.models_initialized or not self.client:
#             return self._fallback_analysis(clean)
#
#         try:
#             result = {
#                 "main_topic": self._extract_main_topic(clean),
#                 "key_points": self._extract_key_points(clean),
#                 "clarity_score": self._clarity_score(clean),
#                 "structure_quality": self._structure_quality(clean),
#                 "problems_detected": self._get_problems_llm(clean),
#                 "llm_analysis": self._get_summary_llm(clean),
#                 "analysis_type": "saiga_llama3_8b_slide"
#             }
#             return result
#         except Exception as e:
#             print(f"[TextAnalyzer] analyze_text exception: {e}")
#             return self._fallback_analysis(clean)
#
#     # ---- LLM wrappers ---------------------------------------------------
#     def _get_summary_llm(self, text: str) -> str:
#         prompt = (
#             "Ты — эксперт по презентациям. Кратко (1-2 предложения) опиши, о чем этот слайд и что главное в нём.\n\n"
#             f"Текст слайда:\n\"{text}\"\n\n"
#             "Краткий вывод:"
#         )
#         return self._call_chat_model(prompt, max_tokens=100, temperature=0.0).strip() or "LLM не дал ответа"
#
#     def _get_problems_llm(self, text: str) -> List[str]:
#         prompt = (
#             "Ты — эксперт по презентациям. Посмотри текст слайда ниже и назови **две основные проблемы**, "
#             "которые мешают восприятию (каждая проблема — в отдельной строке). Не используй JSON и не ставь код-блоки.\n\n"
#             f"{text}\n\n"
#             "Проблемы:\n"
#         )
#         raw = self._call_chat_model(prompt, max_tokens=150, temperature=0.0)
#         return self._parse_lines(raw, default=["Проблемы не определены"])
#
#     # ---- Core LLM call -------------------------------------------------
#     def _call_chat_model(self, user_prompt: str, max_tokens: int = 150, temperature: float = 0.0) -> str:
#         if not self.client:
#             return ""
#         try:
#             response = self.client.chat_completion(
#                 model=self.model_name,
#                 messages=[{"role": "user", "content": user_prompt}],
#                 max_tokens=max_tokens,
#                 temperature=temperature,
#                 top_p=0.9,
#             )
#             # Defensive extraction of text
#             text_out = ""
#             if isinstance(response, dict):
#                 choices = response.get("choices") or response.get("outputs")
#                 if choices and isinstance(choices, list) and len(choices) > 0:
#                     first = choices[0]
#                     msg = first.get("message") or first
#                     if isinstance(msg, dict):
#                         text_out = msg.get("content") or msg.get("text") or ""
#                     else:
#                         text_out = str(first)
#                 else:
#                     text_out = response.get("generated_text", "") or response.get("text", "") or ""
#             else:
#                 text_out = str(response)
#             return self._clean_response(text_out)
#         except Exception as e:
#             print(f"[TextAnalyzer] LLM call error: {e}")
#             return ""
#
#     # ---- Helpers -------------------------------------------------------
#     def _parse_lines(self, text: str, default: List[str]) -> List[str]:
#         if not text:
#             return default
#         # strip fenced code blocks and JSON markers if model still produces them
#         text = re.sub(r'```(?:json)?\s*', '', text)
#         text = re.sub(r'```', '', text)
#         text = re.sub(r'^\s*json\s*[:=]?\s*', '', text, flags=re.IGNORECASE)
#         # split by lines and bullets
#         lines = []
#         for raw in re.split(r'[\n\r]+', text):
#             s = re.sub(r'^[\s\-\d\.\)\:]*', '', raw).strip()
#             if 8 < len(s) <= 200:
#                 lines.append(s)
#         # fallback to sentence splitting
#         if not lines:
#             for s in re.split(r'[.!?]\s+', text):
#                 s = s.strip()
#                 if 8 < len(s) <= 200:
#                     lines.append(s)
#         return lines[:2] if lines else default
#
#     def _clean_text(self, text: str) -> str:
#         t = re.sub(r'\s+', ' ', str(text or "")).strip()
#         return t[:1200]
#
#     def _clean_response(self, text: str) -> str:
#         if not text:
#             return ""
#         text = text.strip()
#         text = re.sub(r'[\x00-\x1f]+', ' ', text)
#         text = re.sub(r'\s+', ' ', text).strip()
#         return text
#
#     # ---- Rule-based utilities -----------------------------------------
#     def _extract_main_topic(self, text: str) -> str:
#         parts = [p.strip() for p in re.split(r'[.!?]', text) if p.strip()]
#         if not parts:
#             return "Тема не определена"
#         first = parts[0]
#         return first if len(first) <= 140 else first[:137] + "..."
#
#     def _extract_key_points(self, text: str) -> List[str]:
#         parts = [p.strip() for p in re.split(r'[.!?]\s*', text) if len(p.strip()) > 10]
#         return parts[:3] if parts else ["Ключевые пункты не определены"]
#
#     def _clarity_score(self, text: str) -> int:
#         words = text.split()
#         sents = [s for s in re.split(r'[.!?]', text) if s.strip()]
#         if not sents:
#             return 3
#         avg_len = len(words) / len(sents)
#         if 10 <= avg_len <= 25:
#             return 8
#         if 5 <= avg_len < 10 or 25 < avg_len <= 40:
#             return 6
#         return 4
#
#     def _structure_quality(self, text: str) -> str:
#         sents = [s for s in re.split(r'[.!?]', text) if s.strip()]
#         if len(sents) >= 4:
#             return "хорошая"
#         if len(sents) >= 2:
#             return "средняя"
#         return "слабая"
#
#     def _fallback_analysis(self, text: str) -> Dict[str, Any]:
#         clean = self._clean_text(text)
#         return {
#             "main_topic": self._extract_main_topic(clean),
#             "key_points": self._extract_key_points(clean),
#             "clarity_score": self._clarity_score(clean),
#             "structure_quality": self._structure_quality(clean),
#             "problems_detected": ["Анализ выполнен без LLM"],
#             "llm_analysis": "LLM не доступна",
#             "analysis_type": "fallback"
#         }
#
#
# # Shared instance
# text_analyzer = TextAnalyzer()
