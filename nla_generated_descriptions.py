#!/usr/bin/env python3
"""Generate with Neuronpedia NLA and write token descriptions to JSON."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import requests


BASE_URL = "https://www.neuronpedia.org"
DEFAULT_MODEL_ID = "llama3.3-70b-it"
DEFAULT_SOURCE_ID = "kitft-l53"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use Neuronpedia's Natural Language Autoencoder API to generate from "
            "a prompt plus assistant prefill, then explain token activations."
        )
    )
    parser.add_argument("--prompt", help="User prompt to send to the model.")
    parser.add_argument(
        "--prompts-file",
        help="Text file containing one prompt per line. Use with --prefills-file or --prefill.",
    )
    parser.add_argument(
        "--prefill",
        help=(
            "Assistant/model prefill text placed before the generated continuation. "
            "In batch mode, this is reused for every prompt unless --prefills-file is provided."
        ),
    )
    parser.add_argument(
        "--prefills-file",
        help="Text file containing one assistant/model prefill per line, paired with --prompts-file.",
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
    parser.add_argument(
        "--describe",
        choices=("generated", "prefill"),
        default="generated",
        help=(
            "Which tokens to describe. Use 'prefill' with --prompts-file to create "
            "first-prefill-token descriptions for each prompt."
        ),
    )
    parser.add_argument(
        "--prefill-token-count",
        type=int,
        default=10,
        help="Number of prefill tokens to explain per prompt when --describe prefill is used.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Only process the first N prompt/prefill pairs. Useful for smoke tests.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="1-based prompt/prefill line number to start from in batch mode.",
    )
    parser.add_argument(
        "--append-output",
        action="store_true",
        help="Load an existing batch output JSON and append new results to it.",
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


def _skip_message_prefix(tokens: list[dict[str, Any]], start: int) -> int:
    while start < len(tokens) and tokens[start].get("token") in {
        "\n",
        "\n\n",
        "<|end_header_id|>",
    }:
        start += 1
    return start


def _assistant_message_starts(tokens: list[dict[str, Any]]) -> list[int]:
    starts: list[int] = []
    for i, token in enumerate(tokens):
        token_text = token.get("token")
        if token_text == "model":
            starts.append(_skip_message_prefix(tokens, i + 1))
        elif token_text == "assistant":
            prev_token = tokens[i - 1].get("token") if i > 0 else None
            if prev_token == "<|start_header_id|>":
                starts.append(_skip_message_prefix(tokens, i + 1))
    return starts


def generated_tokens(tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return tokens after the final assistant/model chat-template marker."""
    assistant_starts = _assistant_message_starts(tokens)
    if assistant_starts:
        return tokens[assistant_starts[-1] :]

    model_indices = [i for i, token in enumerate(tokens) if token.get("token") == "model"]
    if not model_indices:
        raise RuntimeError("Could not find a final assistant/model marker in returned NLA tokens.")

    start = _skip_message_prefix(tokens, model_indices[-1] + 1)
    return tokens[start:]


def assistant_prefill_tokens(tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return tokens in the first assistant/model message, excluding chat-template markers."""
    assistant_starts = _assistant_message_starts(tokens)
    if not assistant_starts:
        raise RuntimeError("Could not find an assistant/model marker in returned NLA tokens.")

    start = assistant_starts[0]
    end = start
    while end < len(tokens):
        token_text = tokens[end].get("token")
        if token_text in {"<end_of_turn>", "<start_of_turn>", "<|eot_id|>", "<|start_header_id|>"}:
            break
        end += 1

    return tokens[start:end]


def read_lines(path: str) -> list[str]:
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_args(args: argparse.Namespace) -> None:
    if args.prefill_token_count < 1:
        raise SystemExit("--prefill-token-count must be at least 1.")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1.")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1 when provided.")
    if args.start_index < 1:
        raise SystemExit("--start-index must be at least 1.")
    if args.append_output and not args.prompts_file:
        raise SystemExit("--append-output is only supported in batch mode.")
    if args.prompts_file:
        if not args.prefill and not args.prefills_file:
            raise SystemExit("Batch mode requires --prefill or --prefills-file.")
        return
    if not args.prompt or not args.prefill:
        raise SystemExit("Single-prompt mode requires --prompt and --prefill.")


def prompt_prefill_pairs(args: argparse.Namespace) -> list[tuple[str, str]]:
    if not args.prompts_file:
        return [(args.prompt, args.prefill)]

    prompts = read_lines(args.prompts_file)
    if args.prefills_file:
        prefills = read_lines(args.prefills_file)
        if len(prompts) != len(prefills):
            raise SystemExit(
                f"{args.prompts_file} has {len(prompts)} prompts but "
                f"{args.prefills_file} has {len(prefills)} prefills."
            )
    else:
        prefills = [args.prefill] * len(prompts)

    return list(zip(prompts, prefills, strict=True))


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


def completion_for_pair(
    *,
    prompt: str,
    prefill: str,
    args: argparse.Namespace,
    request_headers: dict[str, str],
) -> dict[str, Any]:
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": prefill},
    ]
    return post_json(
        "/api/nla/completion",
        {
            "modelId": args.model_id,
            "nlaSourceId": args.source_id,
            "messages": messages,
            "completion_tokens": 1 if args.describe == "prefill" else args.completion_tokens,
            "temperature": args.temperature,
        },
        request_headers,
    )


def tokens_with_descriptions(
    *,
    tokens: list[dict[str, Any]],
    full_text: str,
    args: argparse.Namespace,
    request_headers: dict[str, str],
) -> list[dict[str, Any]]:
    positions = [token["position"] for token in tokens]
    if not positions:
        return []

    explanations = explain_positions(
        full_text=full_text,
        positions=positions,
        args=args,
        request_headers=request_headers,
    )
    by_position = {item["position"]: item for item in explanations}

    return [
        {
            **token,
            "description": by_position[token["position"]]["description"],
            "l2_norm": by_position[token["position"]]["l2_norm"],
            "mse": by_position[token["position"]]["mse"],
            "cosine_similarity": by_position[token["position"]]["cosine_similarity"],
        }
        for token in tokens
    ]


def describe_pair(
    *,
    prompt: str,
    prefill: str,
    args: argparse.Namespace,
    request_headers: dict[str, str],
) -> dict[str, Any]:
    completion_body = completion_for_pair(
        prompt=prompt,
        prefill=prefill,
        args=args,
        request_headers=request_headers,
    )

    if args.describe == "prefill":
        tokens_to_describe = assistant_prefill_tokens(completion_body["tokens"])[
            : args.prefill_token_count
        ]
        output_key = "prefill_tokens"
    else:
        tokens_to_describe = generated_tokens(completion_body["tokens"])
        output_key = "generated_tokens"

    described_tokens = tokens_with_descriptions(
        tokens=tokens_to_describe,
        full_text=completion_body["full_text"],
        args=args,
        request_headers=request_headers,
    )

    output = {
        "modelId": args.model_id,
        "nlaSourceId": args.source_id,
        "prompt": prompt,
        "prefill": prefill,
        "completion": completion_body["completion"],
        "full_text": completion_body["full_text"],
        output_key: described_tokens,
    }
    return output


def batch_output(args: argparse.Namespace, results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "modelId": args.model_id,
        "nlaSourceId": args.source_id,
        "describe": args.describe,
        "prefill_token_count": args.prefill_token_count if args.describe == "prefill" else None,
        "limit": args.limit,
        "count": len(results),
        "results": results,
    }


def write_json(output_path: Path, output: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.tmp")
    temp_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp_path.replace(output_path)


def existing_results(output_path: Path) -> list[dict[str, Any]]:
    if not output_path.exists():
        raise SystemExit(f"--append-output requested but {output_path} does not exist.")

    data = json.loads(output_path.read_text(encoding="utf-8"))
    results = data.get("results")
    if not isinstance(results, list):
        raise SystemExit(f"{output_path} does not look like a batch output JSON file.")
    return results


def main() -> None:
    args = parse_args()
    validate_args(args)
    request_headers = headers(args.api_key)
    output_path = Path(args.output)

    all_pairs = prompt_prefill_pairs(args)
    pairs = all_pairs[args.start_index - 1 :]
    if args.limit is not None:
        pairs = pairs[: max(0, args.limit - args.start_index + 1)]

    results = existing_results(output_path) if args.append_output else []
    if args.append_output and len(results) != args.start_index - 1:
        raise SystemExit(
            f"{output_path} has {len(results)} existing results, but --start-index "
            f"{args.start_index} expects {args.start_index - 1}."
        )

    total = len(all_pairs) if args.limit is None else min(args.limit, len(all_pairs))
    for index, (prompt, prefill) in enumerate(pairs, start=args.start_index):
        if args.prompts_file:
            print(f"Processing {index}/{total}", file=sys.stderr, flush=True)
        results.append(
            describe_pair(prompt=prompt, prefill=prefill, args=args, request_headers=request_headers)
        )
        if args.prompts_file:
            write_json(output_path, batch_output(args, results))

    output: dict[str, Any]
    if args.prompts_file:
        output = batch_output(args, results)
    else:
        output = results[0]

    write_json(output_path, output)
    if args.prompts_file:
        print(f"Wrote {len(results)} prompt/prefill description results to {output_path}")
    else:
        token_key = "prefill_tokens" if args.describe == "prefill" else "generated_tokens"
        print(f"Wrote {len(output[token_key])} {args.describe} token descriptions to {output_path}")


if __name__ == "__main__":
    main()
