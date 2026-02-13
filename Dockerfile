FROM python:3.11-slim

# Install Tesseract + OpenCV system deps
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--timeout", "120", "--workers", "2"]
```

## 4. `requirements.txt`
```
flask==3.0.3
gunicorn==22.0.0
requests==2.32.3
numpy==1.26.4
opencv-python-headless==4.10.0.84
matplotlib==3.9.2
pytesseract==0.3.13
