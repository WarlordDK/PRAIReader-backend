from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, status, Query


from app.schemas import AddDocumentsRequest
from utils import pdf_reader
from utils.all_text_analyzer import AllTextAnalyzer
from utils.content_analyzer import ContentAnalyzer
from utils.image_analyzer import ImageAnalyzer
from utils.rag_analyzer import rag_analyzer
from core.config import get_llm_models_list, get_vlm_models_list
import os
import asyncio

router = APIRouter(prefix="/api", tags=["Анализатор презентаций"])


@router.on_event("startup")
async def startup_event():
    rag_analyzer.initialize()

def _filter_slides_by_flags(slides_text, first_slide: bool, last_slide: bool):
    if not slides_text:
        return [], []

    first_num = slides_text[0]['slide_number']
    last_num = slides_text[-1]['slide_number']

    excluded = set()
    if not first_slide:
        excluded.add(first_num)
    if not last_slide:
        excluded.add(last_num)

    included = [s for s in slides_text if s['slide_number'] not in excluded]
    return included, sorted(list(excluded))

@router.get('/models_llm',
            summary='Все LLM-модели',
            description='Получение списка всех LLM-моделей')
async def get_all_llm_models() -> List[dict]:
    return get_llm_models_list()

@router.get('/models_vlm',
            summary='Все VLM-модели',
            description='Получение списка всех VLM-моделей')
async def get_all_vlm_models() -> List[dict]:
    return get_vlm_models_list()

@router.get('/model_vlm/{model_id}',
            summary='VLM-модели по ID',
            description='Для получения конкретной VLM-модели задайте её ID')
async def get_vlm_model(model_id : int, models = Depends(get_all_vlm_models)):
    for model in models:
        if model.get('id') == model_id : return model
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Указанной vlm-модели не существует")

@router.get('/model_llm/{model_id}',
            summary='LLM-модели по ID',
            description='Для получения конкретной LLM-модели задайте её ID')
async def get_llm_model(model_id : int, models = Depends(get_all_llm_models)):
    for model in models:
        if model.get('id') == model_id : return model
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Указанной llm-модели не существует")

@router.post('/analyze/structure',
             summary='Структурный анализ',
             description='Анализируется количество текста, удобочитаемость, последовательность изложения и т.п.')
async def analyze_presentation(
    file : UploadFile = File(..., description='Загрузите презентацию в формате PDF'),
    model_id: int = Query(1, description='ID LLM-модели'),
    use_rag: bool = Query(False, description='Использование RAG-системы'),
    user_context: str = Query(None, max_length=255, description='Контекст для RAG (промт)'),
    first_slide: bool = Query(True, description='Включение первого слайда в анализ'),
    last_slide: bool = Query(True, description='Включение последнего слайда в анализ'),
    max_tokens: int = Query(2000, gt=300, le=2000, description='Максимальное количество токенов для одного ответа'),
    temperature: float = Query(0.0, ge=0.0, lt=1.0, description='Параметр степени случайности/креативности ответа'),
    models = Depends(get_all_llm_models)
) -> dict:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    model_name = None
    for model in models:
        if model.get('id') == model_id : model_name = model.get('model_name')
    if not model_name:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Модель не найдена')

    try:
        pdf_path = pdf_reader.save_temp_pdf(file)
        slides_text = pdf_reader.extract_text_by_slides(pdf_path)

        included_slides, excluded_slide_numbers = _filter_slides_by_flags(slides_text, first_slide, last_slide)

        full_text_blocks = []
        for slide in included_slides:
            idx = slide.get("slide_number", "?")
            text = slide.get("text", "").strip()
            full_text_blocks.append(f"--- SLIDE {idx} ---\n{text}")

        full_text = "\n\n".join(full_text_blocks)
        rag_output = "rag-система не использовалась"

        if use_rag and user_context:
            relevant_docs = rag_analyzer.query(user_context, top_k=3)
            context_text = "\n".join([d["text"] for d in relevant_docs])
            prompt_with_context = f"{context_text}\n\n{full_text}"
            rag_output = rag_analyzer.query(prompt_with_context)
        else:
            prompt_with_context = full_text

        all_text_analyzer = AllTextAnalyzer(model_name=model_name, max_tokens=max_tokens, temperature=temperature)
        await all_text_analyzer.initialize_models()
        result = all_text_analyzer.analyze_full_text(prompt_with_context)

        os.unlink(pdf_path)

        return {
            "filename": file.filename,
            "total_slides": len(slides_text),
            "excluded_slides": excluded_slide_numbers,
            "summary_report": result,
            "rag_info": rag_output
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/analyze/content",
             summary='Анализ контента',
             description='Анализируется смысловая нагрузка, делается выкладка со всей презентации')
async def analyze_content(
    file : UploadFile = File(..., description='Загрузите презентацию в формате PDF'),
    model_id: int = Query(1, description='ID LLM-модели'),
    first_slide: bool = Query(True, description='Включение первого слайда в анализ'),
    last_slide: bool = Query(True, description='Включение последнего слайда в анализ'),
    max_tokens: int = Query(2000, gt=300, le=2000, description='Максимальное количество токенов для одного ответа'),
    temperature: float = Query(0.0, ge=0.0, lt=1.0, description='Параметр степени случайности/креативности ответа'),
    models = Depends(get_all_llm_models)
) -> dict:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    model_name = None
    for model in models:
        if model.get('id') == model_id : model_name = model.get('model_name')
    if not model_name:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Модель не найдена')

    try:
        pdf_path = pdf_reader.save_temp_pdf(file)
        slides_text = pdf_reader.extract_text_by_slides(pdf_path)

        included_slides, excluded_slide_numbers = _filter_slides_by_flags(slides_text, first_slide, last_slide)

        full_text_blocks = []
        for slide in included_slides:
            idx = slide.get("slide_number", "?")
            text = slide.get("text", "").strip()
            full_text_blocks.append(f"--- SLIDE {idx} ---\n{text}")

        full_text = "\n\n".join(full_text_blocks)

        content_analyzer = ContentAnalyzer(model_name=model_name, max_tokens=max_tokens, temperature=temperature)
        await content_analyzer.initialize_models()
        analysis = content_analyzer.analyze_full_content(full_text)

        os.unlink(pdf_path)

        return {
            "filename": file.filename,
            "total_slides": len(slides_text),
            "excluded_slides": excluded_slide_numbers,
            "content_analysis": analysis
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Content analysis failed: {e}")

@router.post("/analyze/visual",
             summary='Визуальный анализ',
             description='Анализируется заболоченность текста/изображений на слайде')
async def analyze_visual(
        file: UploadFile = File(...),
        model_id: int = Query(1, description='ID VLM-модели'),
        models = Depends(get_all_vlm_models)
) -> dict:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        pdf_path = pdf_reader.save_temp_pdf(file)

        slide_images = pdf_reader.pdf_to_images(pdf_path)

        model_name = None
        for model in models:
            if model.get('id') == model_id: model_name = model.get('model_name')
        if not model_name:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Модель не найдена')

        image_analyzer = ImageAnalyzer(model_name=model_name)
        await image_analyzer.initialize_models()
        result = await image_analyzer.analyze_visual_presentation(slide_images)

        result['strengths'] = result.pop('visual_strengths')
        result['weaknesses'] = result.pop('visual_weaknesses')

        os.unlink(pdf_path)

        return {
            "filename": file.filename,
            "total_slides": len(slide_images),
            "visual_report": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/add",
             summary='Дополнение RAG-системы контекстом',
             description='Добавление новых документов в коллекцию RAG (Qdrant)')
def add_documents_to_rag(data: AddDocumentsRequest) -> dict:
    try:
        if not rag_analyzer.initialized:
            rag_analyzer.initialize()

        rag_analyzer.add_documents(
            docs=data.documents,
            ids=data.ids
        )

        return {
            "status": "success",
            "added": len(data.documents)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))