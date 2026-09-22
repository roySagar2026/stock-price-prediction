# ── Stage 1: Python backend ───────────────────────────────────────────────────
FROM python:3.11-slim AS backend

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY api/      ./api/
COPY config/   ./config/

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]


# ── Stage 2: React frontend (build) ───────────────────────────────────────────
FROM node:20-alpine AS frontend-build

WORKDIR /frontend

COPY frontend/package.json ./
RUN npm install

COPY frontend/ ./
RUN npm run build


# ── Stage 3: Nginx serves React, proxies /api → FastAPI ──────────────────────
FROM nginx:1.25-alpine AS production

COPY --from=frontend-build /frontend/dist /usr/share/nginx/html
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
