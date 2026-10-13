# Image for the HTTP API or a long-running poller. For AWS Lambda see deploy/template.yaml.
FROM python:3.12-slim AS build
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --prefix=/install ".[all]"

FROM python:3.12-slim
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=build /install /usr/local
COPY knowledge ./knowledge
COPY data ./data
USER app
ENV PYTHONUNBUFFERED=1 METRICS_DB=/home/app/.mailpilot/metrics.sqlite3
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"
# 0.0.0.0 is needed inside a container; the API still requires MAILPILOT_API_KEY.
CMD ["mailpilot", "--json-logs", "serve", "--host", "0.0.0.0", "--port", "8000"]
