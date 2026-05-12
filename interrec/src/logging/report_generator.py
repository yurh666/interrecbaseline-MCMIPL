from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.io import read_json
from src.utils.time import now_iso


def generate_run_report(run_dir: str | Path, cfg: dict[str, Any]) -> str:
    run_dir = Path(run_dir)
    summary_path = run_dir / "run_summary.json"
    summary = read_json(summary_path) if summary_path.exists() else {}
    run_id = summary.get("run_id", run_dir.name)
    aggregate = summary.get("aggregate_metrics", {})
    impl_modes = cfg.get("_impl_modes", {})

    mock_list = [(k, v) for k, v in impl_modes.items() if "mock" in str(v).lower()]
    real_list = [(k, v) for k, v in impl_modes.items() if "mock" not in str(v).lower()]

    _mock_impact = {
        "llm": "LLM not called — direction translation & hypothesis are rule-based templates; results DO NOT reflect paper method quality",
        "direction_translator": "Directions use item IDs only — no semantic LLM translation",
        "hypothesis_generator": "Hypotheses from fixed rules — no LLM reasoning",
        "option_writer": "Option text is templated, not LLM-polished",
        "user_simulator": "User choice is deterministic argmax",
        "item_encoder": "Random embeddings — retrieval results are meaningless",
    }

    lines: list[str] = []

    # ── Prominent mock banner ─────────────────────────────────────────────────
    if mock_list:
        lines += [
            f"> # ⚠️  WARNING: THIS RUN USED MOCK COMPONENTS",
            f">",
            f"> **These results are for development / debug purposes only.**",
            f"> **Do NOT use for paper reporting without replacing mock modules.**",
            f">",
        ]
        for k, v in mock_list:
            desc = _mock_impact.get(k, f"mode={v}")
            lines.append(f"> - **`{k}`** is MOCK: {desc}")
        lines += [
            f">",
            f"> See `MOCK_WARNING.md` in the run directory for the full checklist.",
            f"",
        ]

    lines += [
        f"# InterRec Run Report",
        f"",
        f"**run_id:** `{run_id}`",
        f"**timestamp:** {summary.get('timestamp', now_iso())}",
        f"**dataset:** {cfg.get('dataset', {}).get('name', 'unknown')}",
        f"**method:** interrec",
        f"**seed:** {cfg.get('seed', 42)}",
        f"",
        f"## Component Implementation Status",
        f"",
        f"| Module | Status | Notes |",
        f"|--------|--------|-------|",
    ]
    for k, v in mock_list:
        desc = _mock_impact.get(k, "")
        lines.append(f"| `{k}` | ⚠️ **MOCK** | {desc} |")
    for k, v in real_list:
        lines.append(f"| `{k}` | ✅ real (`{v}`) | — |")

    lines += [
        f"",
        f"## Data Split",
        f"- observed_ratio: {cfg.get('dataset', {}).get('observed_ratio', 0.4)} (40%)",
        f"- future_train_ratio: {cfg.get('dataset', {}).get('future_train_ratio', 0.7)}",
        f"- future_valid_ratio: {cfg.get('dataset', {}).get('future_valid_ratio', 0.1)}",
        f"- future_test_ratio: {cfg.get('dataset', {}).get('future_test_ratio', 0.2)}",
        f"",
        f"## Core Metrics",
        f"",
    ]

    if mock_list:
        lines += [
            f"> ⚠️ The metrics below were produced with mock modules.",
            f"> They reflect pipeline correctness, NOT method effectiveness.",
            f"",
        ]

    lines += [
        f"| Metric | Value |",
        f"|--------|-------|",
    ]
    for k, v in aggregate.items():
        if k not in ("method", "dataset", "seed"):
            lines.append(f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |")

    lines += [
        f"",
        f"## Checklist: To get paper-quality results",
        f"",
        f"- [ ] Set `llm.mode: openai` and provide `OPENAI_API_KEY` in environment",
        f"- [ ] Set `embedding.mode: bge` (requires GPU with ≥8GB VRAM)",
        f"- [ ] Set `simulation.user_choice_mode: stochastic_sample`",
        f"- [ ] Use `configs/shared_experiment_config.yaml` for cross-method alignment",
        f"- [ ] Run multiple seeds (0, 1, 2) and report mean ± std",
        f"",
        f"## Files in this run directory",
        f"",
        f"| File | Description |",
        f"|------|-------------|",
        f"| `MOCK_WARNING.md` | Mock component checklist (generated at run start) |",
        f"| `full_log.jsonl` | Per-turn detailed log (includes `impl_modes` field) |",
        f"| `metrics.csv` | Aggregate metrics |",
        f"| `metrics_by_turn.csv` | Turn-level metrics |",
        f"| `config.yaml` | Full configuration used for this run |",
        f"| `environment.txt` | Python / package versions |",
    ]

    report = "\n".join(lines)
    report_path = run_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")
    return str(report_path)
