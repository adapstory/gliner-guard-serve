import os

import pandas as pd
from dotenv import load_dotenv
from locust import FastHttpUser, constant_throughput, task

load_dotenv()

DATASET = os.getenv("DATASET", "prompts")
PROMPTS_FILE = f"{DATASET}.csv" if DATASET != "prompts" else "prompts.csv"
RESPONSES_FILE = "responses.csv"

prompts = pd.read_csv(PROMPTS_FILE)
responses = pd.read_csv(RESPONSES_FILE) if os.path.exists(RESPONSES_FILE) else prompts

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
        response_text = row["assistant_msg"]
        self.client.post(
            "/predict",
            json={
                "text": response_text,
            },
        )
