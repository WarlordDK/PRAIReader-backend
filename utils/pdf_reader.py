import tempfile
from typing import List, Dict

import pymupdf
from pdf2image import convert_from_path
import os

POPPLER_PATH = r"D:\poppler\Library\bin"

def save_temp_pdf(upload_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        tmp.write(upload_file.file.read())
        return tmp.name

def extract_text(pdf_path):
    text = ""
    try:
        doc = pymupdf.open(pdf_path)
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        print(f"Error extracting text: {e}")
    return text

def pdf_to_images(pdf_path):
    try:
        # # Указываем явный путь тулзы poppler, без docker-контейнера
        # images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)

        #При использовании docker-контейнера, тулза автоматически подключается
        images = convert_from_path(pdf_path)
        return images
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return []

def extract_text_by_slides(pdf_path: str) -> List[Dict]:
    slides_text = []
    try:
        doc = pymupdf.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            slides_text.append({
                'slide_number' : page_num + 1,
                'text' : text,
                'word_count' : len(text.split())
            })
        doc.close()
    except Exception as e:
        print(f'Error extracting text by slides : {e}')
    return slides_text