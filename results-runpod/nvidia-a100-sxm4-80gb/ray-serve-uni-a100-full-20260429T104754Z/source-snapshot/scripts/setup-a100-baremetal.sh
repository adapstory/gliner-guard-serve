#!/usr/bin/env bash
# Prepare a bare-metal Runpod/A100 host for Ray Serve benchmarks.
#
# This intentionally does not use Docker or Docker Compose. It mirrors the
# ray-serve Dockerfile setup with uv-managed Python environments and generated
# gRPC stubs copied into Ray's venv so Ray Serve proxy actors can import them.
set -Eeuo pipefail

export PATH="$HOME/.local/bin:$PATH"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

cd "$(dirname "$0")/.."

UV_PYTHON="${UV_PYTHON:-3.13}"
HF_HOME="${HF_HOME:-/workspace/hf-cache}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"

install_curl_if_needed() {
    if command -v curl >/dev/null 2>&1; then
        return 0
    fi
    if command -v apt-get >/dev/null 2>&1; then
        apt-get update
        apt-get install -y curl ca-certificates
        return 0
    fi
    echo "curl is required to install uv and apt-get is unavailable" >&2
    exit 1
}

install_uv_if_needed() {
    if command -v uv >/dev/null 2>&1; then
        return 0
    fi
    install_curl_if_needed
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
}

generate_grpc_stubs() {
    echo "Generating gRPC stubs with uv Python ${UV_PYTHON}..."
    (
        cd ray-serve
        UV_PYTHON="${UV_PYTHON}" uv run python -m grpc_tools.protoc \
            -I=proto --python_out=. --grpc_python_out=. \
            proto/gliner_guard.proto

        site_packages="$(UV_PYTHON="${UV_PYTHON}" uv run python -c 'import site; print(site.getsitepackages()[0])')"
        cp -f gliner_guard_pb2.py gliner_guard_pb2_grpc.py "${site_packages}/"
        echo "Copied stubs to ${site_packages}"
    )
}

mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

install_uv_if_needed

echo "Using uv: $(uv --version)"
UV_PYTHON="${UV_PYTHON}" uv python install "${UV_PYTHON}"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found; GPU verification skipped" >&2
fi

echo "Syncing Ray Serve environment..."
(
    cd ray-serve
    UV_PYTHON="${UV_PYTHON}" uv sync --frozen --no-dev
)

echo "Syncing Locust environment..."
(
    cd test-script
    UV_PYTHON="${UV_PYTHON}" uv sync --frozen --no-dev
)

generate_grpc_stubs

echo "Bare-metal A100 benchmark environment is ready."
