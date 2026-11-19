"""Download a small, fixed HotpotQA sample (dev set, distractor config) for evaluation.

python -m evals.hotpotqa --per-type 3
"""

import argparse
import json
import random
from pathlib import Path

import httpx

ROWS_URL = "https://datasets-server.huggingface.co/rows"
DATASET = {"dataset": "hotpotqa/hotpot_qa", "config": "distractor", "split": "validation"}
DEFAULT_PATH = Path(__file__).parent / "data" / "hotpotqa_sample.json"


def fetch_row(client: httpx.Client, offset: int) -> dict:
    r = client.get(ROWS_URL, params={**DATASET, "offset": offset, "length": 1})
    r.raise_for_status()
    body = r.json()
    row = body["rows"][0]["row"]
    return {
        "id": row["id"],
        "question": row["question"],
        "answer": row["answer"],
        "type": row["type"],
        "level": row["level"],
        "supporting_titles": sorted(set(row["supporting_facts"]["title"])),
    }


def sample(per_type: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    picked: dict[str, list[dict]] = {"bridge": [], "comparison": []}
    with httpx.Client(timeout=30) as client:
        total = client.get(ROWS_URL, params={**DATASET, "offset": 0, "length": 1}).json()["num_rows_total"]
        while any(len(v) < per_type for v in picked.values()):
            row = fetch_row(client, rng.randrange(total))
            if len(picked[row["type"]]) < per_type:
                picked[row["type"]].append(row)
    return picked["bridge"] + picked["comparison"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-type", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()

    rows = sample(args.per_type, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"saved {len(rows)} questions to {args.out}")


if __name__ == "__main__":
    main()
