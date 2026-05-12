#!/usr/bin/env python3
"""Step 5: Temporal-split artifacts for MCMIPL BOOK / MOVIE (InterRec-aligned).

Produces MCMIPL-readable JSON + pickles so BOOK and MOVIE reuse the **same**
chronological protocol as InterRec:

- Profile (conversation-visible history): ``observed_history + future_train``
  (order preserved; dedupe without reordering inside each segment—see merge).
- Test targets for evaluation tuples: ``future_test`` → ``UI_data/test.pkl``.
- Training episode targets (not used as profile): ``future_train`` and
  ``future_valid``, each wrapped as ``[[item_id], ...]`` → ``train.pkl``.

This replaces only ``UI_Interaction_data/review_dict_*.json`` and
``UI_data/{train,test}.pkl`` under ``--dataset-data-dir``.
**Item side metadata** (``fea_item/*.pkl``) is untouched; rerun MCMIPL
``graph_init.py`` afterward so KG users match the new JSON, then retrain/embed.

Supported datasets (**only these two**, per workflow request):

- ``book``: ``BookGraph`` reads users from ``review_dict_valid.json``; RL train
  uses ``valid``, RL test uses ``test``.
- ``movie``: ``MovieGraph`` reads users from ``review_dict_test.json``; RL still
  uses ``valid`` (train) / ``test`` (eval) — export writes **consistent**
  profiles to ``valid``, ``test``, and ``train`` JSON.

Usage::

    cd /home/yurh/interrec
    python scripts/export_interrec_sessions_to_mcmipl_book_movie.py \\
        --dataset book \\
        --sessions data/processed/sessions_book.json \\
        --dataset-data-dir /path/to/MCMIPL/data/book
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any


def _dedupe_ordered(ids: list[str]) -> list[int]:
    seen: set[str] = set()
    out: list[int] = []
    for x in ids:
        if x in seen:
            continue
        seen.add(x)
        out.append(int(x))
    return out


def merge_profile(sess: dict[str, Any]) -> list[int]:
    observed = sess.get("observed_history") or []
    fut_tr = sess.get("future_train") or []
    return _dedupe_ordered([str(i) for i in observed] + [str(i) for i in fut_tr])


def train_pickles(sess: dict[str, Any]) -> list[list[int]] | None:
    rows: list[list[int]] = []
    for part in ("future_train", "future_valid"):
        for x in sess.get(part) or []:
            rows.append([int(x)])
    return rows or None


def test_items(sess: dict[str, Any]) -> list[int] | None:
    xs = sess.get("future_test") or []
    if not xs:
        return None
    return [int(x) for x in xs]


def main() -> None:
    p = argparse.ArgumentParser(description="InterRec sessions → MCMIPL BOOK/MOVIE split files")
    p.add_argument(
        "--dataset",
        choices=("book", "movie"),
        required=True,
    )
    p.add_argument(
        "--sessions",
        type=Path,
        required=True,
        help="InterRec processed sessions.json list",
    )
    p.add_argument(
        "--dataset-data-dir",
        type=Path,
        required=True,
        help="MCMIPL data folder, e.g. .../MCMIPL/data/book",
    )
    args = p.parse_args()

    raw = json.loads(args.sessions.read_text())
    if not isinstance(raw, list):
        raise SystemExit("sessions.json must be a JSON list")

    ui_dir = args.dataset_data_dir / "UI_Interaction_data"
    pkl_dir = args.dataset_data_dir / "UI_data"
    ui_dir.mkdir(parents=True, exist_ok=True)
    pkl_dir.mkdir(parents=True, exist_ok=True)

    review_profile: dict[str, list[int]] = {}
    train_multi: dict[str, list[list[int]]] = {}
    test_multi: dict[str, list[int]] = {}

    skipped_no_profile = 0
    skipped_no_train = 0
    skipped_no_test = 0
    for sess in raw:
        uid = str(sess.get("user_id", "")).strip()
        if not uid:
            continue

        profile = merge_profile(sess)
        if not profile:
            skipped_no_profile += 1
            continue

        tr = train_pickles(sess)
        ts = test_items(sess)
        if tr is None:
            skipped_no_train += 1
            continue
        if ts is None:
            skipped_no_test += 1
            continue

        review_profile[uid] = profile
        train_multi[uid] = tr
        test_multi[uid] = ts

    if not review_profile:
        raise SystemExit("No users exported; check sessions content and thresholds.")

    def dump_json(path: Path, obj: dict[str, Any]) -> None:
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")

    dump_json(ui_dir / "review_dict_train.json", review_profile)
    dump_json(ui_dir / "review_dict_valid.json", review_profile)
    dump_json(ui_dir / "review_dict_test.json", review_profile)

    with (pkl_dir / "train.pkl").open("wb") as f:
        pickle.dump(train_multi, f, protocol=4)
    with (pkl_dir / "test.pkl").open("wb") as f:
        pickle.dump(test_multi, f, protocol=4)

    print(f"[export] dataset={args.dataset}")
    print(f"[export] sessions file: {args.sessions}")
    print(f"[export] users with profile + train.pkl + test.pkl: {len(review_profile)}")
    print(
        "[export] skipped users: "
        f"empty_profile={skipped_no_profile} "
        f"no_train_targets={skipped_no_train} "
        f"no_test_targets={skipped_no_test}"
    )
    print(f"[export] wrote {ui_dir / 'review_dict_*.json'} and {pkl_dir / '*.pkl'}")
    print()
    print("Next: cd MCMIPL && rerun graph construction for BOOK/MOVIE, then TransE + RL.")
    if args.dataset == "movie":
        print("(MovieGraph reads users from review_dict_test.json — file updated accordingly.)")


if __name__ == "__main__":
    main()
