#!/usr/bin/env python3
"""Get top SAE features for prefill text in paired jailbreak/prefill prompts."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests


API_URL = "https://www.neuronpedia.org/api/search-topk-by-token"
MODEL_ID = "gpt-oss-20b"
DEFAULT_LAYER = 15
SOURCE_TEMPLATE = "{layer}-resid-post-aa"
MARKER = "\n\n<<<PREFILL_START>>>\n"


def read_nonempty_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def token_positions_after_marker(tokens: list[str]) -> list[int]:
    text = "".join(tokens)
    marker_start = text.find(MARKER)
    if marker_start == -1:
        raise ValueError("Could not find prefill marker in tokenized response.")

    prefill_char_start = marker_start + len(MARKER)
    positions: list[int] = []
    cursor = 0
    for position, token in enumerate(tokens):
        next_cursor = cursor + len(token)
        if next_cursor > prefill_char_start:
            positions.append(position)
        cursor = next_cursor
    return positions


def feature_description(feature: dict[str, Any]) -> str | None:
    explanations = feature.get("explanations") or []
    if explanations:
        return explanations[0].get("description")
    return None


def request_topk(
    *,
    api_key: str,
    source: str,
    text: str,
    topk_per_token: int,
    density_threshold: float,
    timeout: int,
) -> dict[str, Any]:
    response = requests.post(
        API_URL,
        headers={"x-api-key": api_key, "Content-Type": "application/json"},
        json={
            "modelId": MODEL_ID,
            "source": source,
            "text": text,
            "numResults": topk_per_token,
            "ignoreBos": True,
            "densityThreshold": density_threshold,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def aggregate_prefill_features(payload: dict[str, Any], final_top_n: int) -> tuple[list[str], list[dict[str, Any]]]:
    results = payload.get("results") or []
    tokens = [row.get("token", "") for row in results]
    prefill_positions = set(token_positions_after_marker(tokens))
    features: dict[tuple[str, str, str], dict[str, Any]] = {}
    activation_sums: dict[tuple[str, str, str], float] = defaultdict(float)
    hit_counts: dict[tuple[str, str, str], int] = defaultdict(int)

    for row in results:
        position = row.get("position")
        if position not in prefill_positions:
            continue

        token = row.get("token", "")
        for item in row.get("topFeatures") or []:
            feature = item.get("feature") or {}
            layer = str(feature.get("layer") or payload.get("source"))
            index = str(feature.get("index") or item.get("featureIndex"))
            model_id = str(feature.get("modelId") or MODEL_ID)
            key = (model_id, layer, index)
            activation = float(item.get("activationValue") or 0.0)

            current = features.get(key)
            if current is None or activation > current["max_activation"]:
                features[key] = {
                    "model_id": model_id,
                    "layer": layer,
                    "index": index,
                    "max_activation": activation,
                    "max_activation_token": token,
                    "max_activation_position": position,
                    "description": feature_description(feature),
                    "neuronpedia_url": f"https://www.neuronpedia.org/{model_id}/{layer}/{index}",
                }

            activation_sums[key] += activation
            hit_counts[key] += 1

    top_features = sorted(
        features.values(),
        key=lambda row: (row["max_activation"], activation_sums[(row["model_id"], row["layer"], row["index"])]),
        reverse=True,
    )[:final_top_n]

    for row in top_features:
        key = (row["model_id"], row["layer"], row["index"])
        row["activation_sum"] = activation_sums[key]
        row["prefill_token_hits"] = hit_counts[key]

    prefill_tokens = [results[position].get("token", "") for position in sorted(prefill_positions)]
    return prefill_tokens, top_features


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find top layer-15 Neuronpedia SAE features for paired jailbreak/prefill texts."
    )
    parser.add_argument("--prefills", type=Path, default=Path("prefills.txt"))
    parser.add_argument("--jailbreaks", type=Path, default=Path("jailbreaks.txt"))
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--source", default=None, help="Override the Neuronpedia SAE source ID.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--topk-per-token", type=int, default=20)
    parser.add_argument("--density-threshold", type=float, default=-1)
    parser.add_argument("--sleep", type=float, default=0.1, help="Seconds to sleep between API requests.")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for quick test runs.")
    parser.add_argument("--api-key", default=os.environ.get("NEURONPEDIA_API_KEY"))
    args = parser.parse_args()

    if not args.api_key:
        print("Set NEURONPEDIA_API_KEY or pass --api-key.", file=sys.stderr)
        return 2

    source = args.source or SOURCE_TEMPLATE.format(layer=args.layer)
    output_path = args.output or Path(f"prefill_top{args.top_n}_sae_features_layer{args.layer}.json")

    prefills = read_nonempty_lines(args.prefills)
    jailbreaks = read_nonempty_lines(args.jailbreaks)
    if len(prefills) != len(jailbreaks):
        print(
            f"Expected paired files with the same number of nonempty lines; "
            f"got {len(prefills)} prefills and {len(jailbreaks)} jailbreaks.",
            file=sys.stderr,
        )
        return 2

    pairs = list(zip(jailbreaks, prefills, strict=True))
    if args.limit is not None:
        pairs = pairs[: args.limit]

    output: dict[str, Any] = {
        "model_id": MODEL_ID,
        "source": source,
        "layer": args.layer,
        "aggregation": "max activation over prefill tokens only",
        "items": [],
    }

    for idx, (jailbreak, prefill) in enumerate(pairs, start=1):
        full_text = f"{jailbreak}{MARKER}{prefill}"
        payload = request_topk(
            api_key=args.api_key,
            source=source,
            text=full_text,
            topk_per_token=args.topk_per_token,
            density_threshold=args.density_threshold,
            timeout=args.timeout,
        )
        prefill_tokens, top_features = aggregate_prefill_features(payload, args.top_n)
        output["items"].append(
            {
                "line": idx,
                "jailbreak": jailbreak,
                "prefill": prefill,
                "prefill_tokens": prefill_tokens,
                "top_features": top_features,
            }
        )
        print(f"{idx}/{len(pairs)}: {len(top_features)} features for {prefill!r}", file=sys.stderr)
        if args.sleep and idx < len(pairs):
            time.sleep(args.sleep)

    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
