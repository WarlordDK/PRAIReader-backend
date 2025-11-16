from fastapi import APIRouter, UploadFile, File, HTTPException
from utils import pdf_reader
from utils.all_text_analyzer import all_text_analyzer
from utils.content_analyzer import content_analyzer
import os
import asyncio

router = APIRouter(prefix="/api", tags=["Analyze Presentations"])


@router.on_event("startup")
async def startup_event():
    await asyncio.gather(
        all_text_analyzer.initialize_models(),
        content_analyzer.initialize_models()
    )

@router.get("/")
async def root_info():
    return {
        "message": "PRAIReader (Full-Text Analyzer Only)",
        "full_text_model_ready": all_text_analyzer.models_initialized
    }


@router.post("/analyze/structure")
async def analyze_presentation(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        pdf_path = pdf_reader.save_temp_pdf(file)

        slides_text = pdf_reader.extract_text_by_slides(pdf_path)

        full_text_blocks = []
        for slide in slides_text:
            idx = slide.get("slide_number", "?")
            text = slide.get("text", "").strip()
            full_text_blocks.append(f"--- SLIDE {idx} ---\n{text}")

        full_text = "\n\n".join(full_text_blocks)

        summary_report = all_text_analyzer.analyze_full_text(full_text)

        os.unlink(pdf_path)

        return {
            "filename": file.filename,
            "total_slides": len(slides_text),
            "summary_report": summary_report
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")



@router.post("/analyze/content")
async def analyze_content(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        pdf_path = pdf_reader.save_temp_pdf(file)
        slides_text = pdf_reader.extract_text_by_slides(pdf_path)

        full_text_blocks = []
        for slide in slides_text:
            idx = slide.get("slide_number", "?")
            text = slide.get("text", "").strip()
            full_text_blocks.append(f"--- SLIDE {idx} ---\n{text}")

        full_text = "\n\n".join(full_text_blocks)

        analysis = content_analyzer.analyze_full_content(full_text)

        os.unlink(pdf_path)

        return {
            "filename": file.filename,
            "total_slides": len(slides_text),
            "content_analysis": analysis
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Content analysis failed: {e}")