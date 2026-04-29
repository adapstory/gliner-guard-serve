import logging
import os

import torch
from ray import serve
from ray.serve.config import gRPCOptions

from gliner2 import GLiNER2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_ID = os.environ.get("MODEL_ID", "hivetrace/gliner-guard-uniencoder")
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "0"))
BATCH_WAIT_TIMEOUT = float(os.environ.get("BATCH_WAIT_TIMEOUT", "0.05"))
MAX_CONCURRENT_BATCHES = int(os.environ.get("MAX_CONCURRENT_BATCHES", "1"))
MAX_ONGOING_REQUESTS = int(os.environ.get("MAX_ONGOING_REQUESTS", "200"))
NUM_REPLICAS = int(os.environ.get("NUM_REPLICAS", "1"))
NUM_GPUS_PER_REPLICA = float(os.environ.get("NUM_GPUS_PER_REPLICA", "1"))
NUM_CPUS_PER_REPLICA = float(os.environ.get("NUM_CPUS_PER_REPLICA", "1"))

PII_LABELS = ["person", "address", "email", "phone"]
SAFETY_LABELS = ["safe", "unsafe"]


def _deployment_options() -> dict:
    return {
        "num_replicas": NUM_REPLICAS,
        "max_ongoing_requests": MAX_ONGOING_REQUESTS,
        "ray_actor_options": {
            "num_gpus": NUM_GPUS_PER_REPLICA,
            "num_cpus": NUM_CPUS_PER_REPLICA,
        },
    }


def _to_response(result):
    """Convert model result dict to PredictResponse proto."""
    import gliner_guard_pb2

    entities_map = {}
    for label, values in result.get("entities", {}).items():
        entities_map[label] = gliner_guard_pb2.EntityList(values=values)
    return gliner_guard_pb2.PredictResponse(
        entities=entities_map,
        safety=result.get("safety", ""),
    )


def _build_grpc_deployment():
    """Build gRPC deployment with optional batching."""

    if MAX_BATCH_SIZE > 0:

        @serve.deployment(
            **_deployment_options(),
        )
        class GLiNERGuardGrpcBatched:
            def __init__(self):
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model = GLiNER2.from_pretrained(MODEL_ID)
                self.model.to(self.device).to(torch.bfloat16).eval()
                self.schema = (
                    self.model.create_schema()
                    .entities(entity_types=PII_LABELS, threshold=0.4)
                    .classification(task="safety", labels=SAFETY_LABELS)
                )
                logger.info(
                    (
                        "gRPC model=%s device=%s batch_size=%d timeout=%.3f "
                        "max_concurrent_batches=%d replicas=%d gpu_per_replica=%.3f "
                        "cpu_per_replica=%.3f max_ongoing=%d ready"
                    ),
                    MODEL_ID,
                    self.device,
                    MAX_BATCH_SIZE,
                    BATCH_WAIT_TIMEOUT,
                    MAX_CONCURRENT_BATCHES,
                    NUM_REPLICAS,
                    NUM_GPUS_PER_REPLICA,
                    NUM_CPUS_PER_REPLICA,
                    MAX_ONGOING_REQUESTS,
                )

            @serve.batch(
                max_batch_size=MAX_BATCH_SIZE,
                batch_wait_timeout_s=BATCH_WAIT_TIMEOUT,
                max_concurrent_batches=MAX_CONCURRENT_BATCHES,
            )
            async def _handle_batch(self, requests: list) -> list:
                texts = [req.text for req in requests]
                results = self.model.batch_extract(
                    texts=texts, schemas=self.schema, batch_size=len(texts),
                )
                return [_to_response(r) for r in results]

            async def Predict(self, request) -> "PredictResponse":  # noqa: N802, F821
                return await self._handle_batch(request)

            async def __call__(self, request):
                """REST fallback — used by health checks and REST clients."""
                body = await request.json()
                text = body["text"]
                result = self.model.extract(text, self.schema)
                return result

        return GLiNERGuardGrpcBatched

    @serve.deployment(
        **_deployment_options(),
    )
    class GLiNERGuardGrpc:
        def __init__(self):
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = GLiNER2.from_pretrained(MODEL_ID)
            self.model.to(self.device).to(torch.bfloat16).eval()
            self.schema = (
                self.model.create_schema()
                .entities(entity_types=PII_LABELS, threshold=0.4)
                .classification(task="safety", labels=SAFETY_LABELS)
            )
            logger.info(
                (
                    "gRPC model=%s device=%s replicas=%d gpu_per_replica=%.3f "
                    "cpu_per_replica=%.3f max_ongoing=%d no-batch ready"
                ),
                MODEL_ID,
                self.device,
                NUM_REPLICAS,
                NUM_GPUS_PER_REPLICA,
                NUM_CPUS_PER_REPLICA,
                MAX_ONGOING_REQUESTS,
            )

        async def Predict(self, request) -> "PredictResponse":  # noqa: N802, F821
            result = self.model.extract(request.text, self.schema)
            return _to_response(result)

        async def __call__(self, request):
            """REST fallback — used by health checks and REST clients."""
            body = await request.json()
            text = body["text"]
            result = self.model.extract(text, self.schema)
            return result

    return GLiNERGuardGrpc


DeploymentClass = _build_grpc_deployment()
app = DeploymentClass.bind()

if __name__ == "__main__":
    serve.start(
        http_options={"host": "0.0.0.0", "port": 8000},
        grpc_options=gRPCOptions(
            port=9000,
            grpc_servicer_functions=[
                "gliner_guard_pb2_grpc.add_GLiNERGuardServiceServicer_to_server",
            ],
        ),
    )
    serve.run(app, route_prefix="/predict")
    import signal
    signal.pause()
