"""
从官方 MCMIPL `RL_evaluate.py` 打印的训练日志中提取 eval 指标。

关键区别（易错）：
  - 「批次行」：... reward:x Total epoch_uesr:N  —— 仅为最近 observe_num 个用户的滑动统计，不能代表整次 eval。
  - 「整体均值行」：... reward:x 且行尾无 Total epoch_uesr —— 一次完整 eval（4000 / 2500 用户）后的真实均值，与源码中 best!!! 更新逻辑一致。
"""

from __future__ import annotations

import re
from pathlib import Path

METRICS = ["SR5", "SR10", "SR15", "AvgT", "Rank", "reward"]

# 单行即完整均值；排除训练采样行 rewards:（复数）
_MEAN_LINE = re.compile(
    r"^SR5:([\d.]+), SR10:([\d.]+), SR15:([\d.]+), "
    r"AvgT:([\d.]+), Rank:([\d.]+), reward:([\d.\-]+)\s*$"
)

_LOG_HEADER_STEPS = re.compile(r"\|\s*steps=(\d+)\s*\|")


def parse_log_text(text: str) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for raw in text.replace("\r", "\n").split("\n"):
        if "rewards:" in raw or "Total epoch_uesr" in raw:
            continue
        m = _MEAN_LINE.match(raw.strip())
        if m:
            out.append({k: float(v) for k, v in zip(METRICS, m.groups())})
    return out


def extract_evals(log_path: Path) -> list[dict[str, float]]:
    if not log_path.exists():
        return []
    blob = log_path.read_bytes().decode("utf-8", errors="ignore")
    return parse_log_text(blob)


def best_checkpoint_by_sr15(evals: list[dict[str, float]]) -> dict[str, float] | None:
    """与 RL_model.train 一致：历次 eval 的 SR15 中取最大（官方用 SR15_best 追踪）。"""
    if not evals:
        return None
    return max(evals, key=lambda x: x["SR15"])


def eval_training_step_labels(n_evals: int, eval_num: int = 10) -> list[int]:
    """RL_model：若 eval_num==10，先有 step 0 的初始 eval，再在 10,20,... 触发。"""
    if n_evals <= 0:
        return []
    steps = [0]
    train_evals = n_evals - 1
    for k in range(1, train_evals + 1):
        steps.append(k * eval_num)
    return steps[:n_evals]


def parse_max_training_steps(log_path: Path) -> int | None:
    if not log_path.exists() or log_path.stat().st_size == 0:
        return None
    first_line = log_path.read_bytes().decode(
        "utf-8", errors="ignore"
    ).split("\n", 1)[0]
    m = _LOG_HEADER_STEPS.search(first_line)
    if m:
        return int(m.group(1))
    return None


def count_completed_training_steps(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    text = log_path.read_bytes().decode("utf-8", errors="ignore").replace("\r", "\n")
    return len(re.findall(r"(?m)^loss : [\d.]+ in epoch_uesr \d+$", text))
