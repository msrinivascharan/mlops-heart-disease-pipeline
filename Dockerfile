FROM python:3.11-slim
       WORKDIR /app
       COPY requirements.txt .
       RUN pip install --no-cache-dir -r requirements.txt
       COPY scripts/serve.py .
       COPY models/ ./models/
       EXPOSE 8000
       CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]