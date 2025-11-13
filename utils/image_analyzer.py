import torch
from transformers import pipeline
from typing import Dict, Any, List
from PIL import Image
from core.config import get_hf_token
import asyncio


class ImageAnalyzer:
    def __init__(self):
        self.hf_token = get_hf_token()
        self.model = None
        self.models_initialized = False

    async def initialize_models(self):
        """Инициализация модели для анализа изображений"""
        if self.models_initialized:
            return

        print("🔄 Инициализация image модели...")

        try:
            # Для анализа изображений можно использовать модель для описания изображений
            # Но для простоты пока используем технический анализ
            self.models_initialized = True
            print("✅ Image анализатор инициализирован")

        except Exception as e:
            print(f"❌ Ошибка инициализации image модели: {e}")
            self.models_initialized = False

    def analyze_image(self, image: Image) -> Dict[str, Any]:
        """Анализ изображения слайда"""
        try:
            width, height = image.size
            aspect_ratio = width / height

            return {
                "slide_dimensions": f"{width}x{height}",
                "aspect_ratio": round(aspect_ratio, 2),
                "resolution_quality": self._assess_resolution(width, height),
                "layout_type": self._get_layout_type(aspect_ratio),
                "visual_score": self._calculate_visual_score(width, height),
                "technical_recommendations": self._get_technical_recommendations(width, height, aspect_ratio),
                "design_assessment": self._assess_design(width, height),
                "analysis_type": "technical"
            }

        except Exception as e:
            return {"error": f"Анализ изображения не удался: {str(e)}"}

    def _assess_resolution(self, width: int, height: int) -> str:
        if width >= 1920 and height >= 1080:
            return "высокое (Full HD)"
        elif width >= 1280 and height >= 720:
            return "хорошее (HD)"
        elif width >= 1024 and height >= 768:
            return "стандартное"
        else:
            return "низкое"

    def _get_layout_type(self, aspect_ratio: float) -> str:
        if 1.7 <= aspect_ratio <= 1.8:
            return "16:9 (современный)"
        elif 1.3 <= aspect_ratio <= 1.4:
            return "4:3 (классический)"
        elif aspect_ratio < 1.0:
            return "вертикальный"
        else:
            return "нестандартный"

    def _calculate_visual_score(self, width: int, height: int) -> int:
        if width >= 1920 and height >= 1080:
            return 9
        elif width >= 1280 and height >= 720:
            return 7
        elif width >= 1024 and height >= 768:
            return 6
        else:
            return 4

    def _get_technical_recommendations(self, width: int, height: int, aspect_ratio: float) -> List[str]:
        recommendations = []

        if width < 1280 or height < 720:
            recommendations.append("Увеличьте разрешение для лучшего качества")

        if aspect_ratio < 1.3:
            recommendations.append("Используйте горизонтальную ориентацию для презентаций")

        return recommendations if recommendations else ["Технические параметры в норме"]

    def _assess_design(self, width: int, height: int) -> str:
        if width >= 1280 and height >= 720:
            return "соответствует современным стандартам"
        else:
            return "требует оптимизации"


# Глобальный экземпляр
image_analyzer = ImageAnalyzer()