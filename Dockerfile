FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY frontend ./frontend

ENV POKERION_FRONTEND=/app/frontend
ENV POKERION_DB=/data/pokerion.db

ARG GIT_SHA=dev
ENV POKERION_GIT_SHA=${GIT_SHA}

ENV POKERION_SECURE_COOKIES=1

VOLUME /data
EXPOSE 8000

CMD ["uvicorn", "pokerion.server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
