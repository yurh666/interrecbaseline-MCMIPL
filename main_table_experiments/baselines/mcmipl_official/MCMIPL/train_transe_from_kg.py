#!/usr/bin/env python3
"""Train TransE on GPU from MCMIPL ``kg.pkl`` and write OpenKE-compatible ``transe.pkl``.

The upstream README points to external OpenKE; this script reproduces the same
artifact layout expected by ``RL/env_multi_choice_question.py``:

- ``ui_emb``: shape ``(user.value_len + item.value_len, dim)``
- ``feature_emb``: shape ``(feature.value_len, dim)``
- Row order matches ``construct_graph.py`` global indexing:
  users ``0..U-1``, items as ``U + item_id``, features as ``U + I + fid``.

Run from the ``MCMIPL`` directory (same as ``graph_init.py`` / ``RL_model.py``).

Example::

  export LD_LIBRARY_PATH=.../site-packages/nvidia/cublas/lib:...
  python train_transe_from_kg.py --data_name BOOK --epochs 200 --device cuda
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import utils as U

# (h, r, t) use reciprocal pairs so undirected edges are learnable under TransE.
REL_INTERACT, REL_INTERACT_INV = 0, 1
REL_FRIEND, REL_FRIEND_INV = 2, 3
REL_LIKE, REL_LIKE_INV = 4, 5
REL_BELONG, REL_BELONG_INV = 6, 7
N_RELATIONS = 8


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_triples(kg, n_user: int, n_item: int, n_feat: int) -> List[Tuple[int, int, int]]:
    """Map original ids into global entity ids: user u -> u; item i -> n_user+i; feat f -> n_user+n_item+f."""
    Uoff, Ioff, Foff = 0, n_user, n_user + n_item
    triples: List[Tuple[int, int, int]] = []

    for u_raw, row in kg.G["user"].items():
        u = int(u_raw)
        for it in row["interact"]:
            i = int(it) + Ioff
            triples.append((u, REL_INTERACT, i))
            triples.append((i, REL_INTERACT_INV, u))
        for fr in row["friends"]:
            v = int(fr)
            triples.append((u, REL_FRIEND, v))
            triples.append((v, REL_FRIEND_INV, u))
        for fe in row["like"]:
            f = int(fe) + Foff
            triples.append((u, REL_LIKE, f))
            triples.append((f, REL_LIKE_INV, u))

    for i_raw, row in kg.G["item"].items():
        i = int(i_raw) + Ioff
        for fe in row["belong_to"]:
            f = int(fe) + Foff
            triples.append((i, REL_BELONG, f))
            triples.append((f, REL_BELONG_INV, i))

    n_ent = n_user + n_item + n_feat
    for h, r, t in triples:
        if not (0 <= h < n_ent and 0 <= t < n_ent):
            raise ValueError(f"Triple ({h},{r},{t}) out of range for n_ent={n_ent}")
        if not (0 <= r < N_RELATIONS):
            raise ValueError(f"Bad relation {r}")

    return triples


def _l2(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps))


class TransE(nn.Module):
    def __init__(self, n_entity: int, n_relation: int, dim: int, p_norm: int = 2):
        super().__init__()
        self.dim = dim
        self.p_norm = p_norm
        self.ent = nn.Embedding(n_entity, dim)
        self.rel = nn.Embedding(n_relation, dim)
        nn.init.uniform_(self.ent.weight, -6 / np.sqrt(dim), 6 / np.sqrt(dim))
        nn.init.uniform_(self.rel.weight, -6 / np.sqrt(dim), 6 / np.sqrt(dim))

    def forward_triple(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        eh = self.ent(h)
        er = self.rel(r)
        et = self.ent(t)
        return (eh + er - et).norm(p=self.p_norm, dim=1)


def train(args: argparse.Namespace) -> None:
    _set_seeds(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，改用 cpu")
        device = torch.device("cpu")

    ds = U.load_dataset(args.data_name)
    kg = U.load_kg(args.data_name)
    n_user, n_item, n_feat = ds.user.value_len, ds.item.value_len, ds.feature.value_len
    n_ent = n_user + n_item + n_feat

    triples = _build_triples(kg, n_user, n_item, n_feat)
    if not triples:
        raise RuntimeError("未从 kg 解析到任何三元组，请确认 graph_init 已生成 kg.pkl")

    h_all = torch.tensor([x[0] for x in triples], dtype=torch.long)
    r_all = torch.tensor([x[1] for x in triples], dtype=torch.long)
    t_all = torch.tensor([x[2] for x in triples], dtype=torch.long)
    n_tr = h_all.size(0)

    model = TransE(n_ent, N_RELATIONS, args.dim, p_norm=2).to(device)
    optim = torch.optim.SGD(model.parameters(), lr=args.lr)

    tmp_root = U.TMP_DIR[args.data_name]
    os.makedirs(os.path.join(tmp_root, "embeds"), exist_ok=True)
    out_path = os.path.join(tmp_root, "embeds", "transe.pkl")

    bs = args.batch_size
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(n_tr)
        losses = []
        model.train()
        for start in tqdm(range(0, n_tr, bs), desc=f"epoch {epoch}/{args.epochs}"):
            idx = perm[start : start + bs]
            h = h_all[idx].to(device)
            r = r_all[idx].to(device)
            t = t_all[idx].to(device)

            model.ent.weight.data.copy_(_l2(model.ent.weight.data))

            neg_h = h.clone()
            neg_t = t.clone()
            m = h.size(0)
            head_corrupt = torch.rand(m, device=device) < 0.5
            rand = torch.randint(0, n_ent, (m,), device=device)
            neg_h = torch.where(head_corrupt, rand, neg_h)
            neg_t = torch.where(~head_corrupt, rand, neg_t)

            pos_d = model.forward_triple(h, r, t)
            neg_d = model.forward_triple(neg_h, r, neg_t)
            loss = F.relu(args.margin + pos_d - neg_d).mean()

            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.item())

        print(f"epoch {epoch} mean loss {float(np.mean(losses)):.6f}")

    model.eval()
    with torch.no_grad():
        E = _l2(model.ent.weight).cpu().numpy()
    ui_emb = np.concatenate([E[:n_user], E[n_user : n_user + n_item]], axis=0).astype(np.float32)
    feature_emb = E[n_user + n_item :].astype(np.float32)

    expect_ui = n_user + n_item
    expect_f = n_feat
    if ui_emb.shape != (expect_ui, args.dim) or feature_emb.shape != (expect_f, args.dim):
        raise RuntimeError(
            f"transe 形状与 dataset.pkl 不一致: got ui {ui_emb.shape} feature {feature_emb.shape}, "
            f"期望 ui {(expect_ui, args.dim)} feature {(expect_f, args.dim)}"
        )

    with open(out_path, "wb") as f:
        pickle.dump({"ui_emb": ui_emb, "feature_emb": feature_emb}, f, protocol=4)
    print(f"已写入 {out_path}  ui_emb={ui_emb.shape} feature_emb={feature_emb.shape}")


def main() -> None:
    p = argparse.ArgumentParser(description="Train TransE from kg.pkl (GPU PyTorch)")
    p.add_argument("--data_name", type=str, required=True, choices=list(U.TMP_DIR.keys()))
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--margin", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda", choices=("cuda", "cpu"))
    train(p.parse_args())


if __name__ == "__main__":
    main()
