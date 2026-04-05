import os

import pandas as pd
from dotenv import load_dotenv
from locust import FastHttpUser, constant_throughput, task

load_dotenv()

DATASET = os.getenv("DATASET", "prompts")
PROMPTS_FILE = f"{DATASET}.csv"
# Match responses file to dataset (prompts-short → responses-short, etc.)
RESPONSES_FILE = (
    DATASET.replace("prompts", "responses") + ".csv"
    if DATASET.startswith("prompts")
    else None
)

prompts = pd.read_csv(PROMPTS_FILE)
_has_responses = RESPONSES_FILE and os.path.exists(RESPONSES_FILE)
responses = pd.read_csv(RESPONSES_FILE) if _has_responses else prompts
RESPONSE_COL = "assistant_msg" if _has_responses else "user_msg"

ENTITY_TYPES = [
    "NAME",
    "ADDRESS",
]


class MLServiceUser(FastHttpUser):
    host = os.getenv("GLINER_HOST", "http://localhost:8000")
    wait_time = constant_throughput(5)

    @task
    def predict_prompt(self):
        row = prompts.sample(n=1).iloc[0]
        prompt_text = row["user_msg"]
        self.client.post(
            "/predict",
            json={
                "text": prompt_text,
            },
        )

    @task
    def predict_response(self):
        row = responses.sample(n=1).iloc[0]
        response_text = row[RESPONSE_COL]
        self.client.post(
            "/predict",
            json={
                "text": response_text,
            },
        )
