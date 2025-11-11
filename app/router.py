from fastapi import APIRouter, UploadFile, File
from services import pdf_reader

router = APIRouter(prefix='/api', tags=['Analyze Presentations'])

@router.get('/')
async def info():
    return {'message' : 'analyze router'}

@router.post('/analyze')
async def analyze_presentation(file: UploadFile = File(...)):
    pdf_path = pdf_reader.save_temp_pdf(file)

    text = pdf_reader.extract_text(pdf_path)
    slides = pdf_reader.pdf_to_images(pdf_path)

    return {
        'filename' : file.filename,
        'text_preview' : text[:1000],
        'count_slides' : len(slides)
    }