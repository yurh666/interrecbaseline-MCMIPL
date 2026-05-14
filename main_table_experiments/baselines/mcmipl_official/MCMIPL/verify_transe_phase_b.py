#!/usr/bin/env python3
"""检查 ``transe.pkl`` 是否与 ``dataset.pkl`` 对齐，供 Phase B / RL 加载。"""

from __future__ import annotations

import argparse
import os
import pickle

import numpy as np

import utils as U

EXPECTED_DIM = 64


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_name", type=str, required=True, choices=list(U.TMP_DIR.keys()))
    p.add_argument(
        "--embed_path",
        type=str,
        default="",
        help="默认: tmp/<slug>/embeds/transe.pkl",
    )
    p.add_argument("--dim", type=int, default=EXPECTED_DIM)
    args = p.parse_args()

    ds = U.load_dataset(args.data_name)
    u, i, f = ds.user.value_len, ds.item.value_len, ds.feature.value_len
    expect_ui = u + i
    path = args.embed_path or os.path.join(U.TMP_DIR[args.data_name], "embeds", "transe.pkl")
    if not os.path.isfile(path):
        raise SystemExit(f"缺少文件: {path}")

    with open(path, "rb") as fp:
        emb = pickle.load(fp)
    if not isinstance(emb, dict) or "ui_emb" not in emb or "feature_emb" not in emb:
        raise SystemExit("pickle 必须是含 ui_emb、feature_emb 的字典")

    ui = np.asarray(emb["ui_emb"])
    fe = np.asarray(emb["feature_emb"])
    if ui.shape != (expect_ui, args.dim):
        raise SystemExit(f"ui_emb 形状 {ui.shape} != 期望 {(expect_ui, args.dim)}")
    if fe.shape != (f, args.dim):
        raise SystemExit(f"feature_emb 形状 {fe.shape} != 期望 {(f, args.dim)}")
    if np.isnan(ui).any() or np.isnan(fe).any():
        raise SystemExit("嵌入含 NaN")
    print(f"OK {args.data_name}: {path}")
    print(f"  ui_emb {ui.shape} feature_emb {fe.shape} dtype={ui.dtype}")


if __name__ == "__main__":
    main()
