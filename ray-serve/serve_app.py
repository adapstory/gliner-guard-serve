import logging
import os

import torch
from ray import serve

from gliner2 import GLiNER2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_ID = os.environ.get("MODEL_ID", "hivetrace/gliner-guard-uniencoder")

PII_LABELS = ["person", "address", "email", "phone"]
SAFETY_LABELS = ["safe", "unsafe"]


@serve.deployment(
    num_replicas=1,
    max_ongoing_requests=int(os.environ.get("MAX_ONGOING_REQUESTS", "200")),
    ray_actor_options={"num_gpus": 1},
)
class GLiNERGuardDeployment:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = GLiNER2.from_pretrained(MODEL_ID)
        self.model.to(self.device).to(torch.bfloat16).eval()
        self.schema = (
            self.model.create_schema()
            .entities(entity_types=PII_LABELS, threshold=0.4)
            .classification(task="safety", labels=SAFETY_LABELS)
        )
        logger.info("model=%s device=%s ready", MODEL_ID, self.device)

    async def __call__(self, request):
        body = await request.json()
        text = body["text"]
        result = self.model.extract(text, self.schema)
        return result


app = GLiNERGuardDeployment.bind()
