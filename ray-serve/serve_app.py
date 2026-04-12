import logging
import os

import torch
from ray import serve

from gliner2 import GLiNER2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_ID = os.environ.get("MODEL_ID", "hivetrace/gliner-guard-uniencoder")
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "0"))
BATCH_WAIT_TIMEOUT = float(os.environ.get("BATCH_WAIT_TIMEOUT", "0.05"))
SCHEMA_MODE = os.environ.get("SCHEMA_MODE", "minimal")  # minimal | full

# ---------------------------------------------------------------------------
# Label taxonomies
# ---------------------------------------------------------------------------

PII_LABELS = ["person", "address", "email", "phone"]
SAFETY_LABELS = ["safe", "unsafe"]

# Extended taxonomy (~56 labels total) for label-scaling experiments
PII_LABELS_FULL = [
    "person", "company", "email", "street", "phone",
    "city", "country", "date_of_birth",
]

ADVERSARIAL_LABELS = [
    "none", "instruction_override", "jailbreak_persona",
    "jailbreak_hypothetical", "data_exfiltration", "jailbreak_roleplay",
]

HARMFUL_LABELS = [
    "none", "dangerous_instructions", "harassment",
    "sexual_content", "violence", "hate_speech", "fraud",
    "pii_exposure", "discrimination", "misinformation", "weapons",
]

INTENT_LABELS = [
    "informational", "conversational", "instructional",
    "adversarial", "creative", "threatening",
]

TOV_LABELS = [
    "neutral", "aggressive", "manipulative", "formal", "distressed",
]


def _build_schema(model):
    """Build schema based on SCHEMA_MODE env var.

    minimal (default): 4 PII entities + safety classification (6 labels)
    full: 8 PII entities + safety + adversarial + harmful + intent + tone (56 labels)

    The schema object is built once at init and cached as a dict for
    batch_extract (avoids re-tokenizing label names on every request).
    """
    schema = model.create_schema()

    if SCHEMA_MODE == "full":
        schema = (
            schema
            .entities(entity_types=PII_LABELS_FULL, threshold=0.5)
            .classification(task="safety", labels=SAFETY_LABELS)
            .classification(task="adversarial", labels=ADVERSARIAL_LABELS, multi_label=True)
            .classification(task="harmful", labels=HARMFUL_LABELS, multi_label=True)
            .classification(task="intent", labels=INTENT_LABELS)
            .classification(task="tone", labels=TOV_LABELS)
        )
    else:
        schema = (
            schema
            .entities(entity_types=PII_LABELS, threshold=0.4)
            .classification(task="safety", labels=SAFETY_LABELS)
        )

    return schema


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
                self.schema = _build_schema(self.model)
                logger.info(
                    "model=%s device=%s schema=%s batch_size=%d timeout=%.3f ready",
                    MODEL_ID,
                    self.device,
                    SCHEMA_MODE,
                    MAX_BATCH_SIZE,
                    BATCH_WAIT_TIMEOUT,
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
            self.schema = _build_schema(self.model)
            logger.info(
                "model=%s device=%s schema=%s no-batch ready",
                MODEL_ID, self.device, SCHEMA_MODE,
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
