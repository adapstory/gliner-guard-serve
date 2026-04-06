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

# Generate gRPC stubs and fix imports for package-based import (proto.*)
RUN uv run python -m grpc_tools.protoc \
    -I=proto --python_out=proto --grpc_python_out=proto \
    proto/gliner_guard.proto \
    && sed -i 's/^import gliner_guard_pb2/from proto import gliner_guard_pb2/' \
       proto/gliner_guard_pb2_grpc.py \
    && cp -r proto/ .venv/lib/python3.13/site-packages/proto/

EXPOSE 8000 9000
CMD ["uv", "run", "python", "serve_app.py"]
