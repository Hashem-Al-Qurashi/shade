"""Side-by-side comparison of abliteration vs activation steering."""

from __future__ import annotations

import json as _json
from dataclasses import asdict

from .benchmark import BenchmarkResult


def format_comparison_table(results: list[BenchmarkResult]) -> str:
    """Format multiple BenchmarkResults as a comparison table."""
    has_gsm8k = any(r.gsm8k_accuracy is not None for r in results)
    has_ppl = any(r.perplexity is not None for r in results)
    name_w = max(len(r.model_name) for r in results)
    name_w = max(name_w, len("Technique"))

    header = f"{'Technique':<{name_w}}  {'Refusals':>10}  {'KL Div':>10}"
    if has_ppl:
        header += f"  {'PPL':>8}"
    if has_gsm8k:
        header += f"  {'GSM8K':>8}"

    sep = "-" * len(header)
    lines = [sep, header, sep]

    for r in results:
        refusal_str = f"{r.refusals}/{r.total_prompts}"
        kl_str = f"{r.kl_divergence:.4f}" if r.kl_divergence > 0 else "-"
        line = f"{r.model_name:<{name_w}}  {refusal_str:>10}  {kl_str:>10}"
        if has_ppl:
            ppl_str = f"{r.perplexity:.2f}" if r.perplexity is not None else "-"
            line += f"  {ppl_str:>8}"
        if has_gsm8k:
            gsm_str = f"{r.gsm8k_accuracy:.1%}" if r.gsm8k_accuracy is not None else "-"
            line += f"  {gsm_str:>8}"
        lines.append(line)

    lines.append(sep)
    return "\n".join(lines)


def format_comparison_json(results: list[BenchmarkResult]) -> str:
    """Format multiple BenchmarkResults as JSON."""
    return _json.dumps({"results": [asdict(r) for r in results]}, indent=2)
