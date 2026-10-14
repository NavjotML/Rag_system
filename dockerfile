FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY rag_system/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rag_system/ ./rag_system/

EXPOSE 8501

CMD ["streamlit", "run", "rag_system/app_persistence.py", "--server.address=0.0.0.0", "--server.port=8501"]