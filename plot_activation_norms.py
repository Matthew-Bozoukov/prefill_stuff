#!/usr/bin/env python3
"""Plot NLA activation norms for prompt/prefill runs as SVG files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create average and per-prompt plots of prefill token activation norms."
    )
    parser.add_argument(
        "--input",
        default="prefill_nla_descriptions_50.json",
        help="Batch JSON file produced by nla_generated_descriptions.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="activation_norm_graphs",
        help="Directory where graph SVGs will be written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Number of prompt/prefill results to plot from the start of the JSON.",
    )
    parser.add_argument(
        "--token-key",
        default="prefill_tokens",
        help="Token list key to read from each result.",
    )
    return parser.parse_args()


def slugify(text: str, fallback: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return (slug[:80] or fallback).strip("-")


def load_results(path: Path, limit: int) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results")
    if not isinstance(results, list):
        raise SystemExit(f"{path} does not contain a top-level results list.")
    return results[:limit]


def token_norms(result: dict[str, Any], token_key: str) -> list[float]:
    tokens = result.get(token_key, [])
    return [float(token["l2_norm"]) for token in tokens if "l2_norm" in token]


def token_labels(result: dict[str, Any], token_key: str) -> list[str]:
    tokens = result.get(token_key, [])
    return [str(token.get("token", "")).replace("\n", "\\n") for token in tokens]


def svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_line_svg(
    *,
    path: Path,
    title: str,
    series: list[tuple[str, list[float], str, float]],
    x_labels: list[str] | None = None,
    y_label: str = "Activation l2_norm",
) -> None:
    width = 1000
    height = 620
    left = 90
    right = 35
    top = 70
    bottom = 115 if x_labels else 70
    plot_width = width - left - right
    plot_height = height - top - bottom

    max_len = max((len(values) for _, values, _, _ in series), default=0)
    values = [value for _, vals, _, _ in series for value in vals]
    if max_len == 0 or not values:
        return

    y_min = min(values)
    y_max = max(values)
    if y_min == y_max:
        y_min -= 1
        y_max += 1
    y_pad = (y_max - y_min) * 0.08
    y_min -= y_pad
    y_max += y_pad

    def x_pos(index: int) -> float:
        denominator = max(max_len - 1, 1)
        return left + (index / denominator) * plot_width

    def y_pos(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="34" text-anchor="middle" font-family="Arial, sans-serif" font-size="22" font-weight="700">{svg_escape(title)}</text>',
        f'<text x="24" y="{top + plot_height / 2}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" transform="rotate(-90 24 {top + plot_height / 2})">{svg_escape(y_label)}</text>',
    ]

    for tick in range(6):
        ratio = tick / 5
        y = top + ratio * plot_height
        value = y_max - ratio * (y_max - y_min)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#374151">{value:.0f}</text>')

    parts.append(f'<line x1="{left}" y1="{top + plot_height}" x2="{width - right}" y2="{top + plot_height}" stroke="#111827"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111827"/>')

    for _label, vals, color, opacity in series:
        if not vals:
            continue
        points = " ".join(f"{x_pos(i):.2f},{y_pos(value):.2f}" for i, value in enumerate(vals))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2" stroke-opacity="{opacity}"/>')
        if len(series) <= 3:
            for i, value in enumerate(vals):
                parts.append(f'<circle cx="{x_pos(i):.2f}" cy="{y_pos(value):.2f}" r="4" fill="{color}" fill-opacity="{opacity}"/>')

    tick_labels = x_labels or [str(i) for i in range(1, max_len + 1)]
    for i, label in enumerate(tick_labels[:max_len]):
        x = x_pos(i)
        parts.append(f'<line x1="{x:.2f}" y1="{top + plot_height}" x2="{x:.2f}" y2="{top + plot_height + 6}" stroke="#111827"/>')
        if x_labels:
            parts.append(f'<text x="{x:.2f}" y="{top + plot_height + 18}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#374151" transform="rotate(-35 {x:.2f} {top + plot_height + 18})">{svg_escape(label)}</text>')
        else:
            parts.append(f'<text x="{x:.2f}" y="{top + plot_height + 22}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#374151">{svg_escape(label)}</text>')

    parts.append(f'<text x="{left + plot_width / 2}" y="{height - 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#111827">Prefill token index</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def plot_average(results: list[dict[str, Any]], token_key: str, output_dir: Path) -> None:
    all_norms = [token_norms(result, token_key) for result in results]
    max_len = max((len(norms) for norms in all_norms), default=0)
    if max_len == 0:
        raise SystemExit(f"No l2_norm values found under token key {token_key!r}.")

    xs: list[int] = []
    means: list[float] = []
    mins: list[float] = []
    maxes: list[float] = []
    counts: list[int] = []

    for index in range(max_len):
        values = [norms[index] for norms in all_norms if index < len(norms)]
        xs.append(index + 1)
        means.append(sum(values) / len(values))
        mins.append(min(values))
        maxes.append(max(values))
        counts.append(len(values))

    write_line_svg(
        path=output_dir / "average_activation_norms.svg",
        title=f"Average Prefill Activation Norms Across {len(results)} Prompts",
        series=[
            ("min", mins, "#93c5fd", 0.65),
            ("max", maxes, "#93c5fd", 0.65),
            ("mean", means, "#2563eb", 1.0),
        ],
    )

    summary_path = output_dir / "average_activation_norms.csv"
    lines = ["token_index,mean_l2_norm,min_l2_norm,max_l2_norm,count"]
    lines.extend(
        f"{x},{mean},{low},{high},{count}"
        for x, mean, low, high, count in zip(xs, means, mins, maxes, counts, strict=True)
    )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_overlay(results: list[dict[str, Any]], token_key: str, output_dir: Path) -> None:
    series = []
    for index, result in enumerate(results, start=1):
        norms = token_norms(result, token_key)
        if norms:
            series.append((str(index), norms, "#2563eb", 0.28))

    write_line_svg(
        path=output_dir / "individual_activation_norms_overlay.svg",
        title=f"Individual Prefill Activation Norms Across {len(results)} Prompts",
        series=series,
    )


def plot_individuals(results: list[dict[str, Any]], token_key: str, output_dir: Path) -> None:
    individual_dir = output_dir / "individual_prompts"
    individual_dir.mkdir(parents=True, exist_ok=True)

    for index, result in enumerate(results, start=1):
        norms = token_norms(result, token_key)
        if not norms:
            continue

        labels = token_labels(result, token_key)
        title_text = result.get("prefill") or result.get("prompt") or f"prompt {index}"
        filename = f"{index:02d}_{slugify(str(title_text), f'prompt-{index}')}.svg"

        write_line_svg(
            path=individual_dir / filename,
            title=f"Prompt {index}: Activation Norms",
            series=[("l2_norm", norms, "#2563eb", 1.0)],
            x_labels=labels,
        )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(Path(args.input), args.limit)
    plot_average(results, args.token_key, output_dir)
    plot_overlay(results, args.token_key, output_dir)
    plot_individuals(results, args.token_key, output_dir)

    print(f"Wrote activation norm graphs for {len(results)} prompts to {output_dir}")


if __name__ == "__main__":
    main()
