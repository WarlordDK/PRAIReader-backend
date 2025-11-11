import tempfile, fitz
from pdf2image import convert_from_path
import os

POPPLER_PATH = r"D:\poppler\Library\bin"


def save_temp_pdf(upload_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='pdf') as tmp:
        tmp.write(upload_file.file.read())
        return tmp.name

def extract_text(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def pdf_to_images(pdf_path):
    images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
    return images