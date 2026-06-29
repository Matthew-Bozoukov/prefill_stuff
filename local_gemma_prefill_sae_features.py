#!/usr/bin/env python3
"""Local Gemma Scope 2 top prefill SAE features for Gemma 3 IT models."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from top_prefill_sae_features import MARKER, read_nonempty_lines


DEFAULT_SOURCE_TEMPLATE = "{layer}-gemmascope-2-res-16k"
DEFAULT_SAE_SUBFOLDER_TEMPLATE = "resid_post_all/layer_{layer}_width_16k_l0_big"


def parse_layers(raw_layers: str) -> list[int]:
    layers: list[int] = []
    for chunk in raw_layers.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_raw, end_raw = chunk.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            step = 1 if end >= start else -1
            layers.extend(range(start, end + step, step))
        else:
            layers.append(int(chunk))
    return layers


def source_for_layer(layer: int, source_template: str) -> str:
    return source_template.format(layer=layer)


def positions_after_marker_from_offsets(text: str, offsets: list[tuple[int, int]]) -> list[int]:
    marker_start = text.find(MARKER)
    if marker_start == -1:
        raise ValueError("Could not find prefill marker in text.")
    prefill_char_start = marker_start + len(MARKER)
    return [idx for idx, (_start, end) in enumerate(offsets) if end > prefill_char_start]


def make_feature_row(
    *,
    model_id: str,
    source_template: str,
    layer: int,
    index: int,
    activation: float,
    token: str,
    position: int,
) -> dict[str, Any]:
    source = source_for_layer(layer, source_template)
    return {
        "model_id": model_id,
        "layer": source,
        "index": str(index),
        "max_activation": activation,
        "max_activation_token": token,
        "max_activation_position": position,
        "description": None,
        "neuronpedia_url": f"https://www.neuronpedia.org/{model_id}/{source}/{index}",
    }


def load_sae_params(
    *,
    layer: int,
    sae_repo: str,
    sae_subfolder_template: str,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    subfolder = sae_subfolder_template.format(layer=layer)
    params_path = hf_hub_download(sae_repo, "params.safetensors", subfolder=subfolder)
    params = load_file(params_path, device="cpu")
    return {
        "w_enc": params["w_enc"].to(device=device, dtype=dtype),
        "b_enc": params["b_enc"].to(device=device, dtype=dtype),
        "b_dec": params["b_dec"].to(device=device, dtype=dtype),
        "threshold": params["threshold"].to(device=device, dtype=dtype),
    }


def encode_topk(
    activations: torch.Tensor,
    *,
    sae: dict[str, torch.Tensor],
    topk: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    all_values = []
    all_indices = []
    for start in range(0, activations.shape[0], chunk_size):
        chunk = activations[start : start + chunk_size].to(sae["w_enc"].device, dtype=sae["w_enc"].dtype)
        hidden_pre = (chunk - sae["b_dec"]) @ sae["w_enc"] + sae["b_enc"]
        hidden = hidden_pre * (hidden_pre > sae["threshold"])
        values, indices = torch.topk(hidden, k=topk, dim=-1)
        all_values.append(values.cpu().float())
        all_indices.append(indices.cpu())
    return torch.cat(all_values, dim=0), torch.cat(all_indices, dim=0)


def device_map_entry(device: torch.device) -> int | str:
    if device.type == "cuda":
        return device.index if device.index is not None else 0
    return device.type


def cache_prefill_activations(
    *,
    pairs: list[tuple[str, str]],
    layers: list[int],
    cache_path: Path,
    model_id: str,
    hf_model_id: str,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_id,
        torch_dtype=dtype,
        device_map={"": device_map_entry(device)},
        attn_implementation="eager",
    )
    model.eval()

    items: list[dict[str, Any]] = []
    layer_activations: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}

    for idx, (jailbreak, prefill) in enumerate(pairs, start=1):
        text = f"{jailbreak}{MARKER}{prefill}"
        encoded_offsets = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        offsets = encoded_offsets["offset_mapping"]
        prefill_positions = positions_after_marker_from_offsets(text, offsets)
        input_ids = torch.tensor([encoded_offsets["input_ids"]], device=device)
        attention_mask = torch.ones_like(input_ids)

        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

        tokens = [tokenizer.decode([token_id]) for token_id in encoded_offsets["input_ids"]]
        item = {
            "line": idx,
            "jailbreak": jailbreak,
            "prefill": prefill,
            "prefill_token_positions": prefill_positions,
            "prefill_tokens": [tokens[position] for position in prefill_positions],
        }
        items.append(item)

        for layer in layers:
            layer_hidden = outputs.hidden_states[layer + 1][0, prefill_positions, :].detach().cpu().to(torch.bfloat16)
            layer_activations[layer].append(layer_hidden)

        del outputs
        print(f"{model_id}: cached activations {idx}/{len(pairs)}", file=sys.stderr, flush=True)

    data = {
        "model_id": model_id,
        "hf_model_id": hf_model_id,
        "layers": layers,
        "items": items,
        "layer_activations": {layer: torch.cat(layer_activations[layer], dim=0) for layer in layers},
        "line_lengths": [len(item["prefill_token_positions"]) for item in items],
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, cache_path)
    return data


def load_or_build_cache(args: argparse.Namespace, pairs: list[tuple[str, str]], layers: list[int]) -> dict[str, Any]:
    if args.cache.exists() and not args.rebuild_cache:
        return torch.load(args.cache, map_location="cpu", weights_only=False)
    return cache_prefill_activations(
        pairs=pairs,
        layers=layers,
        cache_path=args.cache,
        model_id=args.model_id,
        hf_model_id=args.hf_model_id,
        device=torch.device(args.device),
        dtype={"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype],
    )


def write_layer_output(
    *,
    layer: int,
    cache: dict[str, Any],
    values: torch.Tensor,
    indices: torch.Tensor,
    output_path: Path,
    model_id: str,
    source_template: str,
    top_n: int,
) -> None:
    lengths = cache["line_lengths"]
    items_meta = cache["items"]
    cursor = 0
    output: dict[str, Any] = {
        "model_id": model_id,
        "source": source_for_layer(layer, source_template),
        "layer": layer,
        "aggregation": "max activation over prefill tokens only",
        "items": [],
    }

    for item_meta, length in zip(items_meta, lengths, strict=True):
        feature_rows: dict[int, dict[str, Any]] = {}
        activation_sums: defaultdict[int, float] = defaultdict(float)
        hit_counts: defaultdict[int, int] = defaultdict(int)

        for local_pos in range(length):
            global_pos = cursor + local_pos
            token = item_meta["prefill_tokens"][local_pos]
            position = item_meta["prefill_token_positions"][local_pos]
            for value, feature_idx in zip(values[global_pos].tolist(), indices[global_pos].tolist(), strict=True):
                if value <= 0:
                    continue
                current = feature_rows.get(feature_idx)
                if current is None or value > current["max_activation"]:
                    feature_rows[feature_idx] = make_feature_row(
                        model_id=model_id,
                        source_template=source_template,
                        layer=layer,
                        index=feature_idx,
                        activation=value,
                        token=token,
                        position=position,
                    )
                activation_sums[feature_idx] += value
                hit_counts[feature_idx] += 1

        top_features = sorted(
            feature_rows.values(),
            key=lambda row: (row["max_activation"], activation_sums[int(row["index"])]),
            reverse=True,
        )[:top_n]
        for row in top_features:
            feature_idx = int(row["index"])
            row["activation_sum"] = activation_sums[feature_idx]
            row["prefill_token_hits"] = hit_counts[feature_idx]

        output["items"].append(
            {
                "line": item_meta["line"],
                "jailbreak": item_meta["jailbreak"],
                "prefill": item_meta["prefill"],
                "prefill_tokens": item_meta["prefill_tokens"],
                "top_features": top_features,
            }
        )
        cursor += length

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefills", type=Path, default=Path("prefills.txt"))
    parser.add_argument("--jailbreaks", type=Path, default=Path("jailbreaks.txt"))
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--hf-model-id", required=True)
    parser.add_argument("--sae-repo", required=True)
    parser.add_argument("--sae-subfolder-template", default=DEFAULT_SAE_SUBFOLDER_TEMPLATE)
    parser.add_argument("--source-template", default=DEFAULT_SOURCE_TEMPLATE)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--topk-per-token", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if os.environ.get("HUGGINGFACE_API_KEY") and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_API_KEY"]

    layers = parse_layers(args.layers)
    prefills = read_nonempty_lines(args.prefills)
    jailbreaks = read_nonempty_lines(args.jailbreaks)
    if len(prefills) != len(jailbreaks):
        print(f"Expected paired files; got {len(prefills)} prefills and {len(jailbreaks)} jailbreaks.", file=sys.stderr)
        return 2
    pairs = list(zip(jailbreaks, prefills, strict=True))
    if args.limit is not None:
        pairs = pairs[: args.limit]

    cache = load_or_build_cache(args, pairs, layers)
    manifest_items = []
    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    for layer in layers:
        print(f"{args.model_id}: encoding layer {layer}", file=sys.stderr, flush=True)
        sae = load_sae_params(
            layer=layer,
            sae_repo=args.sae_repo,
            sae_subfolder_template=args.sae_subfolder_template,
            device=device,
            dtype=dtype,
        )
        values, indices = encode_topk(
            cache["layer_activations"][layer],
            sae=sae,
            topk=args.topk_per_token,
            chunk_size=args.chunk_size,
        )
        output_path = args.output_dir / f"prefill_top{args.top_n}_sae_features_layer{layer}.json"
        write_layer_output(
            layer=layer,
            cache=cache,
            values=values,
            indices=indices,
            output_path=output_path,
            model_id=args.model_id,
            source_template=args.source_template,
            top_n=args.top_n,
        )
        manifest_items.append(
            {
                "layer": layer,
                "source": source_for_layer(layer, args.source_template),
                "output": str(output_path),
                "count": len(cache["items"]),
            }
        )
        del sae, values, indices
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest_path = args.manifest or (args.output_dir / "manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "model_id": args.model_id,
                "hf_model_id": args.hf_model_id,
                "sae_repo": args.sae_repo,
                "sae_subfolder_template": args.sae_subfolder_template,
                "source_template": args.source_template,
                "layers": layers,
                "prefills": str(args.prefills),
                "jailbreaks": str(args.jailbreaks),
                "limit": args.limit,
                "items": manifest_items,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {manifest_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
