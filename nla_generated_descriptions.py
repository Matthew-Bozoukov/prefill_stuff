#!/usr/bin/env python3
"""Generate with Neuronpedia NLA and write generated-token descriptions to JSON."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import requests


BASE_URL = "https://www.neuronpedia.org"
DEFAULT_MODEL_ID = "gemma-3-27b-it"
DEFAULT_SOURCE_ID = "kitft-l41"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use Neuronpedia's Natural Language Autoencoder API to generate from "
            "a prompt plus assistant prefill, then explain generated token activations."
        )
    )
    parser.add_argument("--prompt", required=True, help="User prompt to send to the model.")
    parser.add_argument(
        "--prefill",
        required=True,
        help="Assistant/model prefill text placed before the generated continuation.",
    )
    parser.add_argument("--output", default="nla_generated_descriptions.json")
    parser.add_argument("--api-key", default=os.environ.get("NEURONPEDIA_API_KEY"))
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--source-id", default=DEFAULT_SOURCE_ID)
    parser.add_argument("--completion-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Max token positions to explain per request. Neuronpedia's demo uses 16.",
    )
    return parser.parse_args()


def headers(api_key: str | None) -> dict[str, str]:
    out = {"Content-Type": "application/json"}
    if api_key:
        out["x-api-key"] = api_key
    return out


def post_json(endpoint: str, payload: dict[str, Any], request_headers: dict[str, str]) -> dict[str, Any]:
    response = requests.post(
        f"{BASE_URL}{endpoint}",
        headers=request_headers,
        json=payload,
        timeout=180,
    )
    try:
        body = response.json()
    except ValueError as exc:
        raise RuntimeError(f"{endpoint} returned non-JSON response: {response.text}") from exc

    if not response.ok:
        raise RuntimeError(
            f"{endpoint} failed with HTTP {response.status_code}: "
            f"{json.dumps(body, ensure_ascii=False)}"
        )

    return body


def generated_tokens(tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return tokens after the final chat-template model marker."""
    model_indices = [i for i, token in enumerate(tokens) if token.get("token") == "model"]
    if not model_indices:
        raise RuntimeError("Could not find a final model marker in returned NLA tokens.")

    start = model_indices[-1] + 1
    while start < len(tokens) and tokens[start].get("token") in {"\n", "\n\n"}:
        start += 1
    return tokens[start:]


def explain_positions(
    *,
    full_text: str,
    positions: list[int],
    args: argparse.Namespace,
    request_headers: dict[str, str],
) -> list[dict[str, Any]]:
    explanations: list[dict[str, Any]] = []
    for start in range(0, len(positions), args.batch_size):
        batch = positions[start : start + args.batch_size]
        body = post_json(
            "/api/nla/explain",
            {
                "modelId": args.model_id,
                "nlaSourceId": args.source_id,
                "text": full_text,
                "positions": batch,
                "temperature": args.temperature,
            },
            request_headers,
        )
        explanations.extend(body["results"])
    return explanations


def main() -> None:
    args = parse_args()
    request_headers = headers(args.api_key)

    messages = [
        {"role": "user", "content": args.prompt},
        {"role": "assistant", "content": args.prefill},
    ]
    completion_body = post_json(
        "/api/nla/completion",
        {
            "modelId": args.model_id,
            "nlaSourceId": args.source_id,
            "messages": messages,
            "completion_tokens": args.completion_tokens,
            "temperature": args.temperature,
        },
        request_headers,
    )

    generated = generated_tokens(completion_body["tokens"])
    positions = [token["position"] for token in generated]
    explanations = explain_positions(
        full_text=completion_body["full_text"],
        positions=positions,
        args=args,
        request_headers=request_headers,
    )

    by_position = {item["position"]: item for item in explanations}
    generated_with_descriptions = [
        {
            **token,
            "description": by_position[token["position"]]["description"],
            "l2_norm": by_position[token["position"]]["l2_norm"],
            "mse": by_position[token["position"]]["mse"],
            "cosine_similarity": by_position[token["position"]]["cosine_similarity"],
        }
        for token in generated
    ]

    output = {
        "modelId": args.model_id,
        "nlaSourceId": args.source_id,
        "prompt": args.prompt,
        "prefill": args.prefill,
        "completion": completion_body["completion"],
        "full_text": completion_body["full_text"],
        "generated_tokens": generated_with_descriptions,
    }

    output_path = Path(args.output)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(generated_with_descriptions)} generated token descriptions to {output_path}")


if __name__ == "__main__":
    main()
