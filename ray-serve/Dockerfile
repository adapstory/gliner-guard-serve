FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON=3.13 \
    HF_HOME=/app/.cache/huggingface \
    RAY_SERVE_HTTP_HOST=0.0.0.0 \
    RAY_SERVE_HTTP_PORT=8000 \
    PYTHONPATH=/app

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . .

# Generate gRPC stubs at root level (flat import: gliner_guard_pb2)
# and copy into site-packages so Ray ProxyActor can import them
RUN uv run python -m grpc_tools.protoc \
    -I=proto --python_out=. --grpc_python_out=. \
    proto/gliner_guard.proto \
    && cp gliner_guard_pb2.py gliner_guard_pb2_grpc.py \
       .venv/lib/python3.13/site-packages/

EXPOSE 8000 9000
CMD ["uv", "run", "python", "serve_app.py"]
