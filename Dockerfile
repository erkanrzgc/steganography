# syntax=docker/dockerfile:1.7
FROM node:22.22.1-bookworm-slim AS web
WORKDIR /src/web
COPY web/package.json web/package-lock.json* ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

FROM python:3.11.13-slim-bookworm AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    STEGANO_STATE_DIR=/state
RUN groupadd --system --gid 10001 steganography \
 && useradd --system --uid 10001 --gid 10001 --home /nonexistent steganography
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY api ./api
COPY core ./core
COPY modules ./modules
COPY report ./report
COPY steganography ./steganography
COPY ui ./ui
COPY cli.py config.py registry.py ./
COPY docker-entrypoint.sh /usr/local/bin/steganography-entrypoint
RUN pip --default-timeout=300 install --retries 10 --no-cache-dir '.[api]'
COPY --from=web /src/web/dist ./steganography/web
RUN chmod 0755 /usr/local/bin/steganography-entrypoint \
 && mkdir -p /state && chown 10001:10001 /state
USER 10001:10001
EXPOSE 8000
VOLUME ["/state"]
ENTRYPOINT ["/usr/local/bin/steganography-entrypoint"]
CMD ["steganography", "--quiet", "serve", "--host", "0.0.0.0", "--port", "8000"]
