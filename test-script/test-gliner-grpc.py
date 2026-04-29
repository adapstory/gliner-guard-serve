"""Locust gRPC load test for GLiNER Guard via Ray Serve."""

import os
import sys
import time

import grpc
import grpc.experimental.gevent as grpc_gevent
import pandas as pd
from dotenv import load_dotenv
from locust import User, constant_throughput, events, task

# Locust runs users as gevent greenlets. Without this grpcio can block the whole
# worker process, which makes RPS plateau while per-call latency looks low.
grpc_gevent.init_gevent()

# Add ray-serve/ to path so generated stubs (flat) are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ray-serve"))

import gliner_guard_pb2  # noqa: E402
import gliner_guard_pb2_grpc  # noqa: E402

load_dotenv()

DATASET = os.getenv("DATASET", "prompts")
USER_THROUGHPUT = float(os.getenv("LOCUST_USER_THROUGHPUT", "5"))
PROMPTS_FILE = f"{DATASET}.csv"
RESPONSES_FILE = (
    DATASET.replace("prompts", "responses") + ".csv"
    if DATASET.startswith("prompts")
    else None
)

prompts = pd.read_csv(PROMPTS_FILE)
_has_responses = RESPONSES_FILE and os.path.exists(RESPONSES_FILE)
responses = pd.read_csv(RESPONSES_FILE) if _has_responses else prompts
RESPONSE_COL = "assistant_msg" if _has_responses else "user_msg"


class GrpcUser(User):
    host = os.getenv("GLINER_HOST", "localhost:9000")
    wait_time = constant_throughput(USER_THROUGHPUT)

    def on_start(self):
        self.channel = grpc.insecure_channel(
            self.host,
            options=[
                ("grpc.max_receive_message_length", 50 * 1024 * 1024),
                ("grpc.keepalive_time_ms", 30000),
            ],
        )
        self.stub = gliner_guard_pb2_grpc.GLiNERGuardServiceStub(self.channel)

    def _predict(self, text: str, name: str) -> None:
        request = gliner_guard_pb2.PredictRequest(text=text)
        start = time.perf_counter()
        try:
            response = self.stub.Predict(request)
            elapsed_ms = (time.perf_counter() - start) * 1000
            events.request.fire(
                request_type="gRPC",
                name=name,
                response_time=elapsed_ms,
                response_length=response.ByteSize(),
                exception=None,
                context={},
            )
        except grpc.RpcError as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            events.request.fire(
                request_type="gRPC",
                name=name,
                response_time=elapsed_ms,
                response_length=0,
                exception=e,
                context={},
            )

    @task
    def predict_prompt(self):
        row = prompts.sample(n=1).iloc[0]
        self._predict(row["user_msg"], "predict_prompt")

    @task
    def predict_response(self):
        row = responses.sample(n=1).iloc[0]
        self._predict(row[RESPONSE_COL], "predict_response")

    def on_stop(self):
        self.channel.close()
