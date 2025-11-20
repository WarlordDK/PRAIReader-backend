from fastapi import APIRouter, UploadFile, File, HTTPException

from app.schemas import AddDocumentsRequest
from utils import pdf_reader
from utils.all_text_analyzer import all_text_analyzer
from utils.content_analyzer import content_analyzer
from utils.image_analyzer import image_analyzer
from utils.rag_analyzer import rag_analyzer
import os
import asyncio

router = APIRouter(prefix="/api", tags=["Analyze Presentations"])


@router.on_event("startup")
async def startup_event():
    await asyncio.gather(
        all_text_analyzer.initialize_models(),
        content_analyzer.initialize_models(),
        image_analyzer.initialize_models(),
    )

    rag_analyzer.initialize()

@router.get("/")
async def root_info():
    return {
        "message": "PRAIReader (Full-Text Analyzer Only)",
        "full_text_model_ready": all_text_analyzer.models_initialized
    }


def _filter_slides_by_flags(slides_text, first_slide: bool, last_slide: bool):
    """
    slides_text: list of dicts {'slide_number': int, 'text': str, ...}
    Возвращает (included_slides, excluded_slide_numbers)
    included_slides — список словарей, которые попадут в анализ (в том же формате),
    excluded_slide_numbers — список номеров исключённых слайдов.
    """
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


@router.post("/analyze/structure")
async def analyze_presentation(
    file: UploadFile = File(...),
    use_rag: bool = False,
    user_context: str = "",
    first_slide: bool = True,
    last_slide: bool = True
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        pdf_path = pdf_reader.save_temp_pdf(file)
        slides_text = pdf_reader.extract_text_by_slides(pdf_path)  # original list with numbers

        # фильтрация
        included_slides, excluded_slide_numbers = _filter_slides_by_flags(slides_text, first_slide, last_slide)

        # Собираем full_text таким образом, чтобы сохранять реальные номера слайдов (--- SLIDE N ---)
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

        result = all_text_analyzer.analyze_full_text(prompt_with_context)

        os.unlink(pdf_path)

        return {
            "filename": file.filename,
            "total_slides": len(slides_text),            # сохраняем общее количество слайдов
            "excluded_slides": excluded_slide_numbers,   # список исключённых номеров (для фронта)
            "summary_report": result,
            "rag_info": rag_output
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/analyze/content")
async def analyze_content(
    file: UploadFile = File(...),
    first_slide: bool = True,
    last_slide: bool = True
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

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

@router.post("/analyze/visual")
async def analyze_visual(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        pdf_path = pdf_reader.save_temp_pdf(file)

        slide_images = pdf_reader.pdf_to_images(pdf_path)


        result = await image_analyzer.analyze_visual_presentation(slide_images)

        os.unlink(pdf_path)

        return {
            "filename": file.filename,
            "total_slides": len(slide_images),
            "visual_report": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/add")
def add_documents_to_rag(data: AddDocumentsRequest):
    """
    Добавление новых документов в коллекцию RAG (Qdrant).
    """
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