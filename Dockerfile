FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    poppler-utils \
    libjpeg-dev \
    libopenjp2-7 \
    libfreetype6 \
    liblcms2-2 \
    libtiff6 \
    libpng16-16 \
    libmupdf-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --prefer-binary PyMuPDF==1.24.8

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
