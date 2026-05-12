#!/usr/bin/env python3
"""
End-to-end mock: InterRec temporal export → MCMIPL BOOK graph_init → env sanity checks.

Writes a fresh copy of MCMIPL (code only) under /tmp, injects minimal fea_item
pickles covering all items in sliced sessions, runs export_interrec_sessions_to_mcmipl_book_movie,
then graph_init BOOK and MultiChoiceRecommendEnv(train/test) reset().

Requires: rsync, and MCMIPL deps (conda env mcmipl-reproduce recommended).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

WORKDIR = Path("/tmp/mcmipl_book_temporal_mock")
SCRIPT_DIR = Path(__file__).resolve().parent
INTERREC_ROOT = SCRIPT_DIR.parent
MCMIPL_SRC = Path("/home/yurh/main_table_experiments/baselines/mcmipl_official/MCMIPL")
EXPORT_PY = SCRIPT_DIR / "export_interrec_sessions_to_mcmipl_book_movie.py"
DEFAULT_SESSIONS = INTERREC_ROOT / "data/processed/sessions.json"


def _patch_test_target_iterable(workdir: Path) -> None:
    """MCMIPL test-reset assumes target_item is iterable; numpy scalars fail."""
    p = workdir / "RL" / "env_multi_choice_question.py"
    t = p.read_text()
    if "ti = self.ui_array[self.test_num, 1]" in t:
        return
    old = (
        "            self.user_id = self.ui_array[self.test_num, 0]\n"
        "            self.target_item  = self.ui_array[self.test_num, 1]\n"
        "            self.test_num += 1\n"
    )
    new = (
        "            self.user_id = int(self.ui_array[self.test_num, 0])\n"
        "            ti = self.ui_array[self.test_num, 1]\n"
        "            self.target_item = [int(ti)] if not isinstance(ti, (list, tuple)) else list(ti)\n"
        "            self.test_num += 1\n"
    )
    if old not in t:
        raise SystemExit("env_multi_choice_question.py patch template mismatch — update mock script")
    p.write_text(t.replace(old, new, 1))


def _collect_items(sessions: list[dict]) -> set[int]:
    keys = ("observed_history", "future_train", "future_valid", "future_test")
    out: set[int] = set()
    for s in sessions:
        for k in keys:
            for x in s.get(k) or []:
                out.add(int(x))
    return out


def _write_minimal_feas(data_book: Path, item_ids: set[int]) -> None:
    """Align with Graph_generate/book_graph.py: item_feature + small_to_large pickles."""
    fea_dir = data_book / "fea_item"
    fea_dir.mkdir(parents=True, exist_ok=True)
    sorted_items = sorted(item_ids)
    # Overlapping small features so KG path always has reachable attrs after init asks
    pool = [f"sf_{j}" for j in range(24)]
    small_to_large: dict[str, tuple[int, ...]] = {k: tuple() for k in pool}
    item_feature: dict[int, list[str]] = {}
    for item in sorted_items:
        h = item % len(pool)
        feats = [pool[h], pool[(h + 3) % len(pool)], pool[(h + 7) % len(pool)]]
        item_feature[item] = feats
    with (fea_dir / "small_to_large.pkl").open("wb") as f:
        import pickle

        pickle.dump(small_to_large, f, protocol=4)
    with (fea_dir / "item_feature.pkl").open("wb") as f:
        pickle.dump(item_feature, f, protocol=4)


def main() -> None:
    if not EXPORT_PY.exists():
        print("Missing export script:", EXPORT_PY)
        sys.exit(1)
    if not MCMIPL_SRC.exists():
        print("Missing MCMIPL:", MCMIPL_SRC)
        sys.exit(1)
    sess_path = DEFAULT_SESSIONS
    sessions = json.loads(sess_path.read_text())
    subset = sessions[:8]
    if len(subset) < 2:
        print("Need ≥2 sessions in", sess_path)
        sys.exit(1)

    slice_path = Path("/tmp/mcmipl_mock_sessions_slice.json")
    slice_path.write_text(json.dumps(subset, ensure_ascii=False, indent=2))

    items = _collect_items(subset)
    if not items:
        print("No items in session slice")
        sys.exit(1)

    if WORKDIR.exists():
        shutil.rmtree(WORKDIR)

    subprocess.run(
        [
            "rsync",
            "-a",
            "--exclude",
            "data",
            "--exclude",
            "tmp",
            "--exclude",
            "__pycache__",
            f"{MCMIPL_SRC}/",
            f"{WORKDIR}/",
        ],
        check=True,
    )

    book_data = WORKDIR / "data" / "book"
    book_data.mkdir(parents=True, exist_ok=True)
    tmp_book = WORKDIR / "tmp" / "book"
    tmp_book.mkdir(parents=True, exist_ok=True)

    _patch_test_target_iterable(WORKDIR)

    _write_minimal_feas(book_data, items)

    subprocess.run(
        [sys.executable, str(EXPORT_PY), "--dataset", "book", "--sessions", str(slice_path), "--dataset-data-dir", str(book_data)],
        check=True,
    )

    subprocess.run([sys.executable, "graph_init.py", "--data_name", "BOOK"], cwd=str(WORKDIR), check=True)

    snippet = """
import os
import pickle
import numpy as np
from utils import BOOK, TMP_DIR, load_kg, load_dataset
from RL.env_multi_choice_question import MultiChoiceRecommendEnv

os.chdir("{cwd}")
kg = load_kg(BOOK)
dataset = load_dataset(BOOK)
u_len, i_len, f_len = dataset.user.value_len, dataset.item.value_len, dataset.feature.value_len
emb_dir = os.path.join(TMP_DIR[BOOK], "embeds")
os.makedirs(emb_dir, exist_ok=True)
rng = np.random.RandomState(0)
dummy = {{
    "ui_emb": rng.randn(u_len + i_len, 64).astype("float32"),
    "feature_emb": rng.randn(f_len, 64).astype("float32"),
}}
with open(os.path.join(emb_dir, "transe.pkl"), "wb") as f:
    pickle.dump(dummy, f, protocol=4)

common_kw = dict(
    max_turn=15, cand_num=10, cand_item_num=10, attr_num=20, ask_num=1,
    entropy_way='weight entropy', fm_epoch=0, choice_num=4,
)
env_tr = MultiChoiceRecommendEnv(
    kg, dataset, BOOK, embed="transe", seed=0, mode="train", **common_kw
)
state, cand, asp = env_tr.reset()
print('TRAIN reset: state_dim', len(state), 'cand', len(cand))

env_te = MultiChoiceRecommendEnv(
    kg, dataset, BOOK, embed="transe", seed=0, mode="test", **common_kw
)
state2, cand2, asp2 = env_te.reset()
print('TEST reset: state_dim', len(state2), 'user', env_te.user_id, 'target_item', env_te.target_item)
ua = env_te.ui_array
if ua is not None:
    print('TEST scheduled ui_pairs', int(ua.shape[0]))
print('MOCK_OK')
""".format(cwd=str(WORKDIR))

    subprocess.run([sys.executable, "-c", snippet], cwd=str(WORKDIR), check=True)
    print()
    print("=== Mock completed: BOOK temporal export + graph_init + env train/test OK ===")


if __name__ == "__main__":
    main()
