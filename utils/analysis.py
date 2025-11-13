from typing import Dict, Any, List
from PIL import Image
from utils.text_analyzer import text_analyzer
from utils.image_analyzer import image_analyzer
import asyncio


class PresentationAnalyzer:
    def __init__(self):
        self.models_initialized = False

    async def initialize_models(self):
        """Инициализация всех моделей"""
        print("🔄 Запуск инициализации всех моделей...")

        # Инициализируем текстовую модель
        await text_analyzer.initialize_models()

        self.models_initialized = text_analyzer.models_initialized
        print(f"✅ Все модели инициализированы: {self.models_initialized}")

    def analyze_slide_content(self, text: str, image: Image = None) -> Dict[str, Any]:
        """Анализ содержимого слайда"""
        # Анализ текста
        text_analysis = text_analyzer.analyze_text(text)

        # Анализ изображения если есть
        visual_analysis = image_analyzer.analyze_image(image) if image else {}

        return {
            "text_analysis": text_analysis,
            "visual_analysis": visual_analysis,
            "overall_score": self._calculate_overall_score(text_analysis, visual_analysis),
            "analysis_type": "enhanced" if self.models_initialized else "basic"
        }

    def _calculate_overall_score(self, text_analysis: Dict, visual_analysis: Dict) -> float:
        """Расчет общей оценки"""
        text_score = text_analysis.get('clarity_score', 5)
        visual_score = visual_analysis.get('visual_score', 5)
        return round((text_score + visual_score) / 2, 1)

    def generate_summary_report(self, slides_analysis: List[Dict]) -> Dict[str, Any]:
        """Генерация итогового отчета"""
        if not slides_analysis:
            return self._get_empty_summary()

        try:
            total_slides = len(slides_analysis)
            total_score = sum(slide.get('analysis', {}).get('overall_score', 5) for slide in slides_analysis)
            avg_score = total_score / total_slides

            # Собираем все рекомендации и проблемы
            all_recommendations = []
            all_problems = []

            for slide in slides_analysis:
                analysis = slide.get('analysis', {})
                text_analysis = analysis.get('text_analysis', {})

                all_recommendations.extend(text_analysis.get('specific_recommendations', []))
                all_problems.extend(text_analysis.get('problems_detected', []))

            # Уникальные элементы
            unique_recommendations = list(set([r for r in all_recommendations if len(r) > 10]))[:4]
            unique_problems = list(set([p for p in all_problems if len(p) > 10]))[:4]

            return {
                "presentation_score": round(avg_score, 1),
                "total_slides_analyzed": total_slides,
                "key_strengths": self._extract_strengths(avg_score),
                "critical_issues": unique_problems if unique_problems else ["Серьезные проблемы не выявлены"],
                "priority_recommendations": unique_recommendations if unique_recommendations else [
                    "Продолжайте в том же духе"],
                "target_audience": self._determine_audience(avg_score),
                "overall_verdict": self._get_verdict(avg_score)
            }

        except Exception as e:
            print(f"Ошибка генерации отчета: {e}")
            return self._get_empty_summary()

    def _extract_strengths(self, avg_score: float) -> List[str]:
        if avg_score >= 7:
            return ["Хорошая структура", "Понятное изложение"]
        elif avg_score >= 5:
            return ["Информативная подача", "Логичное построение"]
        else:
            return ["Потенциал для развития"]

    def _determine_audience(self, avg_score: float) -> str:
        if avg_score >= 8:
            return "Широкая аудитория"
        elif avg_score >= 6:
            return "Общая аудитория"
        else:
            return "Требуется адаптация"

    def _get_verdict(self, avg_score: float) -> str:
        if avg_score >= 8:
            return "Отличная презентация"
        elif avg_score >= 6:
            return "Хорошая основа"
        elif avg_score >= 4:
            return "Требует доработки"
        else:
            return "Необходима переработка"

    def _get_empty_summary(self) -> Dict[str, Any]:
        return {
            "presentation_score": 0,
            "total_slides_analyzed": 0,
            "key_strengths": ["Данные отсутствуют"],
            "critical_issues": ["Анализ не выполнен"],
            "priority_recommendations": ["Загрузите презентацию"],
            "target_audience": "Не определена",
            "overall_verdict": "Анализ не выполнен"
        }


# Глобальный экземпляр
analyzer = PresentationAnalyzer()