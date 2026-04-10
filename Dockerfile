FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip && pip install -e .

ENV PYTHONUNBUFFERED=1
ENV MAX_CONCURRENT_ENVS=64

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
