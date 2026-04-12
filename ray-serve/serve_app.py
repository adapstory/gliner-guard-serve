import logging
import os

import torch
from ray import serve

from gliner2 import GLiNER2
from gliner2.inference.schema_registry import SchemaRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_ID = os.environ.get("MODEL_ID", "hivetrace/gliner-guard-uniencoder")
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "0"))
BATCH_WAIT_TIMEOUT = float(os.environ.get("BATCH_WAIT_TIMEOUT", "0.05"))
SCHEMA_MODE = os.environ.get("SCHEMA_MODE", "minimal")  # minimal | full


def _create_registry() -> SchemaRegistry:
    """Build a SchemaRegistry based on SCHEMA_MODE env var.

    minimal (default): 4 PII entities + safety classification (6 labels)
    full: 8 PII entities + safety + adversarial + harmful + intent + tone (56 labels)

    Returns a registry ready for plugin-driven label registration.
    """
    registry = SchemaRegistry(max_labels=100)

    if SCHEMA_MODE == "full":
        registry.register_entities([
            "person", "company", "email", "street", "phone",
            "city", "country", "date_of_birth",
        ], threshold=0.5)
        registry.register_classification("safety", ["safe", "unsafe"])
        registry.register_classification("adversarial", [
            "none", "instruction_override", "jailbreak_persona",
            "jailbreak_hypothetical", "data_exfiltration", "jailbreak_roleplay",
        ], multi_label=True)
        registry.register_classification("harmful", [
            "none", "dangerous_instructions", "harassment",
            "sexual_content", "violence", "hate_speech", "fraud",
            "pii_exposure", "discrimination", "misinformation", "weapons",
        ], multi_label=True)
        registry.register_classification("intent", [
            "informational", "conversational", "instructional",
            "adversarial", "creative", "threatening",
        ])
        registry.register_classification("tone", [
            "neutral", "aggressive", "manipulative", "formal", "distressed",
        ])
    else:
        registry.register_entities(["person", "address", "email", "phone"], threshold=0.4)
        registry.register_classification("safety", ["safe", "unsafe"])

    return registry


def _build_deployment():
    """Build the appropriate deployment class based on MAX_BATCH_SIZE."""

    if MAX_BATCH_SIZE > 0:

        @serve.deployment(
            num_replicas=1,
            max_ongoing_requests=int(
                os.environ.get("MAX_ONGOING_REQUESTS", "200")
            ),
        )
        class GLiNERGuardBatched:
            def __init__(self):
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model = GLiNER2.from_pretrained(MODEL_ID)
                self.model.to(self.device).to(torch.bfloat16).eval()
                registry = _create_registry()
                self.schema = registry.build_schema(self.model)
                logger.info(
                    "model=%s device=%s schema=%s batch_size=%d timeout=%.3f registry=%s ready",
                    MODEL_ID,
                    self.device,
                    SCHEMA_MODE,
                    MAX_BATCH_SIZE,
                    BATCH_WAIT_TIMEOUT,
                    registry.summary(),
                )

            @serve.batch(
                max_batch_size=MAX_BATCH_SIZE,
                batch_wait_timeout_s=BATCH_WAIT_TIMEOUT,
            )
            async def handle_batch(self, texts: list[str]) -> list[dict]:
                logger.info("batch_extract called with %d texts", len(texts))
                results = self.model.batch_extract(
                    texts=texts,
                    schemas=self.schema,
                    batch_size=len(texts),
                )
                return results

            async def __call__(self, request):
                body = await request.json()
                return await self.handle_batch(body["text"])

        return GLiNERGuardBatched

    @serve.deployment(
        num_replicas=1,
        max_ongoing_requests=int(
            os.environ.get("MAX_ONGOING_REQUESTS", "200")
        ),
    )
    class GLiNERGuardDeployment:
        def __init__(self):
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = GLiNER2.from_pretrained(MODEL_ID)
            self.model.to(self.device).to(torch.bfloat16).eval()
            registry = _create_registry()
            self.schema = registry.build_schema(self.model)
            logger.info(
                "model=%s device=%s schema=%s registry=%s no-batch ready",
                MODEL_ID, self.device, SCHEMA_MODE, registry.summary(),
            )

        async def __call__(self, request):
            body = await request.json()
            text = body["text"]
            result = self.model.extract(text, self.schema)
            return result

    return GLiNERGuardDeployment


DeploymentClass = _build_deployment()
app = DeploymentClass.bind()

if __name__ == "__main__":
    serve.start(http_options={"host": "0.0.0.0", "port": 8000})
    serve.run(app, route_prefix="/predict")
    import signal
    signal.pause()
