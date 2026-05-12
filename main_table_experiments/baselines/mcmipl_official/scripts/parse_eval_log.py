"""Parse official MCMIPL training / evaluate logs into JSON metrics."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CMP = _REPO_ROOT / "comparison"
if _CMP.is_dir():
    sys.path.insert(0, str(_CMP))

try:
    from mcmipl_log_metrics import parse_log_text, best_checkpoint_by_sr15
except ImportError:  # 独立拷贝脚本时降级
    parse_log_text = None  # type: ignore
    best_checkpoint_by_sr15 = None  # type: ignore

_BEST_FALLBACK = re.compile(
    r"best!!!+SR5:(?P<SR5>[\d.\-]+),\s*SR10:(?P<SR10>[\d.\-]+),\s*"
    r"SR15:(?P<SR15>[\d.\-]+),\s*AvgT:(?P<AvgT>[\d.\-]+),\s*Rank:(?P<hDCG>[\d.\-]+)!!!+"
)


def parse_metrics(text: str) -> dict:
    """与主仓库 `comparison/mcmipl_log_metrics.py` 对齐：首选各次 eval 的整体均值中取 SR@15 最优。"""
    if parse_log_text and best_checkpoint_by_sr15:
        best = best_checkpoint_by_sr15(parse_log_text(text))
        if best:
            out: dict[str, float] = {}
            if "SR5" in best:
                out["SR@5"] = best["SR5"]
            if "SR10" in best:
                out["SR@10"] = best["SR10"]
            if "SR15" in best:
                out["SR@15"] = best["SR15"]
            out["AvgT"] = best["AvgT"]
            out["hDCG"] = best["Rank"]
            out["reward"] = best["reward"]
            return out

    hits = list(_BEST_FALLBACK.finditer(text))
    if hits:
        g = hits[-1].groupdict()
        return {
            "SR@5": float(g["SR5"]),
            "SR@10": float(g["SR10"]),
            "SR@15": float(g["SR15"]),
            "AvgT": float(g["AvgT"]),
            "hDCG": float(g["hDCG"]),
        }
    return {}


def main():
    if len(sys.argv) != 7:
        raise SystemExit(
            "Usage: parse_eval_log.py <log> <out_json> <data_name> <seed> <epoch> <exit_code>"
        )

    log_file = Path(sys.argv[1])
    out_file = Path(sys.argv[2])
    data_name = sys.argv[3]
    seed = int(sys.argv[4])
    epoch = int(sys.argv[5])
    exit_code = int(sys.argv[6])

    text = log_file.read_text(encoding="utf-8", errors="replace") if log_file.exists() else ""
    metrics = parse_metrics(text)
    payload = {
        "method": "MCMIPL",
        "dataset": data_name,
        "seed": seed,
        "checkpoint_epoch": epoch,
        "exit_code": exit_code,
        "log_file": str(log_file),
        "metrics_found": bool(metrics),
        **metrics,
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
