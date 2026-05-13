#!/usr/bin/env python3
"""Step 5: Temporal-split artifacts for MCMIPL BOOK / MOVIE / LAST_FM_STAR / YELP_STAR (InterRec-aligned).

Produces MCMIPL-readable JSON + pickles so these datasets reuse the **same**
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

Supported datasets:

- ``book``: ``BookGraph`` reads users from ``review_dict_valid.json``; RL train
  uses ``valid``, RL test uses ``test``.
- ``movie``: ``MovieGraph`` reads users from ``review_dict_test.json``; RL still
  uses ``valid`` (train) / ``test`` (eval) — export writes **consistent**
  profiles to ``valid``, ``test``, and ``train`` JSON.
- ``lastfm_star``: ``LastFmGraph`` reads ``review_dict_train.json``; users must
  appear in both ``Graph_generate_data/user_friends.pkl`` and ``user_like.pkl``
  (export keeps only the intersection).
- ``yelp_star``: KG users come from ``Graph_generate_data/user_dict.json``;
  export keeps only those user ids.

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


def load_allowed_user_ids(dataset: str, dataset_data_dir: Path) -> set[int] | None:
    """Users that must appear in exported UI data for graph_init (KG alignment)."""
    if dataset in ("book", "movie"):
        return None
    gd = dataset_data_dir / "Graph_generate_data"
    if dataset == "lastfm_star":
        with (gd / "user_friends.pkl").open("rb") as f:
            uf = pickle.load(f)
        with (gd / "user_like.pkl").open("rb") as f:
            ul = pickle.load(f)
        kf = {int(k) for k in uf.keys()}
        kl = {int(k) for k in ul.keys()}
        return kf & kl
    if dataset == "yelp_star":
        with (gd / "user_dict.json").open(encoding="utf-8") as f:
            d = json.load(f)
        return {int(k) for k in d.keys()}
    raise ValueError(f"unsupported dataset for KG filter: {dataset}")


_GRAPH_INIT_HINT = {
    "book": "BOOK",
    "movie": "MOVIE",
    "lastfm_star": "LAST_FM_STAR",
    "yelp_star": "YELP_STAR",
}


def main() -> None:
    p = argparse.ArgumentParser(
        description="InterRec sessions → MCMIPL UI split files (BOOK/MOVIE/LAST_FM_STAR/YELP_STAR)"
    )
    p.add_argument(
        "--dataset",
        choices=("book", "movie", "lastfm_star", "yelp_star"),
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

    allowed = load_allowed_user_ids(args.dataset, args.dataset_data_dir)

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
    skipped_not_in_graph = 0
    for sess in raw:
        uid = str(sess.get("user_id", "")).strip()
        if not uid:
            continue
        if allowed is not None and int(uid) not in allowed:
            skipped_not_in_graph += 1
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
        f"no_test_targets={skipped_no_test} "
        f"not_in_kg_entities={skipped_not_in_graph}"
    )
    print(f"[export] wrote {ui_dir / 'review_dict_*.json'} and {pkl_dir / '*.pkl'}")
    print()
    dn = _GRAPH_INIT_HINT[args.dataset]
    print(f"Next: cd MCMIPL && python graph_init.py --data_name {dn}, then TransE + RL.")
    if args.dataset == "movie":
        print("(MovieGraph reads users from review_dict_test.json — file updated accordingly.)")


if __name__ == "__main__":
    main()
