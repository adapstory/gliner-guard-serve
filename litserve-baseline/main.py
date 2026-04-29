import litserve as ls
from gliner2 import GLiNER2
from pydantic import BaseModel
import logging
import os
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import torch

PII_LABELS = ["person", "address", "email", "phone"]
SAFETY_LABELS = ["safe", "unsafe"]
MODEL_ID = os.environ.get(
    "MODEL_ID", os.environ.get("TORCH_MODEL_NAME", "hivetrace/gliner-guard-uniencoder")
)
TORCH_DTYPE = os.environ.get("TORCH_DTYPE", "bfloat16")
MAX_BATCH_SIZE = int(os.environ.get(
    "LITSERVE_MAX_BATCH_SIZE", os.environ.get("MAX_BATCH_SIZE", "64")
))
BATCH_TIMEOUT = float(os.environ.get(
    "LITSERVE_BATCH_TIMEOUT", os.environ.get("BATCH_WAIT_TIMEOUT", "0.05")
))
WORKERS_PER_DEVICE = int(os.environ.get("LITSERVE_WORKERS_PER_DEVICE", "4"))


class GuardRequest(BaseModel):
    text: str


class GLiNERGuardAPI(ls.LitAPI):
    def setup(self, device):
        self.model = GLiNER2.from_pretrained(MODEL_ID)
        self.model.to(device).to(getattr(torch, TORCH_DTYPE))
        self.schema = (
            self.model.create_schema()
            .entities(entity_types=PII_LABELS, threshold=0.4)
            .classification(task="safety", labels=SAFETY_LABELS)
        )
        logger.info("device=%s  max_batch_size=%d", device, self.max_batch_size)

    def decode_request(self, request: GuardRequest):
        return request.text

    def batch(self, inputs):
        return inputs

    def predict(self, batch):
        logger.info("batch_size=%d", len(batch))
        results = self.model.batch_extract(
            texts=batch,
            schemas=self.schema,
            batch_size=len(batch),
        )
        return results

    def unbatch(self, output):
        return output

    def encode_response(self, output):
        return output


if __name__ == "__main__":
    api = GLiNERGuardAPI(max_batch_size=MAX_BATCH_SIZE, batch_timeout=BATCH_TIMEOUT)
    server = ls.LitServer(
        api,
        accelerator="auto",
        timeout=30,
        workers_per_device=WORKERS_PER_DEVICE,
        fast_queue=True,
    )
    server.run(port=8000, generate_client_file=False)
