from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default="https://heart-failure-api-1053145910416.europe-west1.run.app",
    )
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--samples-dir", default="scripts")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def choose_random_sample(samples_dir: Path, seed: int | None) -> Path:
    candidates = sorted(samples_dir.glob("sample_request*.json"))
    if not candidates:
        raise FileNotFoundError(f"no sample_request*.json files found in {samples_dir}")

    rng = random.Random(seed)
    return rng.choice(candidates)


def post_prediction(base_url: str, api_key: str, payload: bytes) -> tuple[int, str]:
    request = Request(
        url=f"{base_url.rstrip('/')}/predict",
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
        },
    )

    try:
        with urlopen(request, timeout=30) as response:
            return response.status, response.read().decode("utf-8")
    except HTTPError as err:
        return err.code, err.read().decode("utf-8")
    except URLError as err:
        raise RuntimeError(f"request failed: {err}") from err


def main() -> None:
    args = parse_args()
    sample_path = choose_random_sample(Path(args.samples_dir), args.seed)

    payload_text = sample_path.read_text(encoding="utf-8")
    payload_json = json.loads(payload_text)
    payload = json.dumps(payload_json).encode("utf-8")

    status_code, response_text = post_prediction(args.base_url, args.api_key, payload)

    print(f"selected sample: {sample_path}")
    print(f"status: {status_code}")

    try:
        parsed = json.loads(response_text)
        print(json.dumps(parsed, indent=2))
    except json.JSONDecodeError:
        print(response_text)


if __name__ == "__main__":
    main()
