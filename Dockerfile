FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN groupadd --system homebuddy && useradd --system --gid homebuddy homebuddy \
    && mkdir -p /data && chown homebuddy:homebuddy /data

COPY pyproject.toml ./
COPY app ./app
COPY scripts ./scripts
COPY simulator-input /input
RUN pip install --upgrade pip && pip install .

USER homebuddy
EXPOSE 18743

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:18743/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "18743", "--proxy-headers"]
