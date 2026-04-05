"""Download and prepare external benchmark datasets.

Outputs CSV files in test-script/ directory:
  - xstest.csv      (XSTest safety benchmark, ~450 rows)
  - aya-rus.csv      (AYA Russian subset, 500 rows)
"""

import csv
import random
from pathlib import Path

from datasets import load_dataset

OUT_DIR = Path(__file__).resolve().parent.parent / "test-script"


def prepare_xstest() -> None:
    ds = load_dataset("walledai/XSTest")
    out = OUT_DIR / "xstest.csv"
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["user_msg"])
        for row in ds["test"]:
            writer.writerow([row["prompt"]])
    print(f"XSTest: {len(ds['test'])} rows → {out}")


def prepare_aya_russian(n: int = 500) -> None:
    ds = load_dataset("CohereForAI/aya_dataset")
    rus = ds["train"].filter(lambda x: x["language_code"] == "rus")
    indices = random.sample(range(len(rus)), min(n, len(rus)))
    out = OUT_DIR / "aya-rus.csv"
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["user_msg"])
        for i in indices:
            writer.writerow([rus[i]["inputs"]])
    print(f"AYA Russian: {len(indices)} rows → {out}")


if __name__ == "__main__":
    prepare_xstest()
    prepare_aya_russian()
