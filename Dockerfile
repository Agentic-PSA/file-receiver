FROM python:3.12-slim

WORKDIR /app

# Disable stdout buffering so logs appear in real-time in kubectl logs
ENV PYTHONUNBUFFERED=1

# Install poppler for PDF-to-image conversion (pdf2image)
RUN apt-get update && \
    apt-get install -y --no-install-recommends poppler-utils && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Storage volume mount point
RUN mkdir -p /data/files

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]