# Hello-world image for the Phase 2 hosting acceptance test. Not the serving image.
#
#     docker build -t protloc-hello deployment/hello
#
# BALLAST_MB adds an incompressible layer so image-pull time can be measured at the
# size of a PyTorch serving image. It is random bytes, which do not compress, so the
# platform pulls as many bytes as it would for real weights. The default is 0.
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN pip install "fastapi==0.141.1" "uvicorn==0.52.4"

ARG BALLAST_MB=0
RUN if [ "${BALLAST_MB}" -gt 0 ]; then head -c "$((BALLAST_MB * 1024 * 1024))" /dev/urandom > /ballast.bin; fi

RUN useradd --create-home --uid 10001 hello
USER hello
WORKDIR /app
COPY --chown=hello:hello main.py .

# Cloud Run injects PORT (8080). `exec` makes uvicorn PID 1 so SIGTERM reaches it.
CMD ["/bin/sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port \"${PORT:-8080}\""]
