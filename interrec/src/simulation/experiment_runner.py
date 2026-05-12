from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.agents.direction_translator import DirectionTranslator
from src.agents.hypothesis_generator import HypothesisGenerator
from src.agents.llm_client import LLMClient
from src.agents.option_writer import OptionWriter
from src.data.dataset import InterRecDataset
from src.embedding.bm25_index import BM25Index
from src.embedding.index import EmbeddingIndex
from src.logging.run_logger import RunLogger
from src.recommendation.recommender import BeliefRecommender
from src.simulation.session_simulator import run_session
from src.simulation.theta_builder import build_theta_star


def build_components(
    cfg: dict[str, Any],
    dataset: InterRecDataset | None = None,
) -> dict[str, Any]:
    """Instantiate all runtime components from config.

    dataset is passed so that DirectionTranslator and HypothesisGenerator can use
    real item metadata (artist names, tags) in LLM prompts instead of raw numeric IDs.
    """
    llm_cfg = cfg.get("llm", {})
    llm = LLMClient(
        mode=llm_cfg.get("mode", "mock"),
        provider=llm_cfg.get("provider", "none"),
        log_prompts=llm_cfg.get("log_prompts", True),
        log_responses=llm_cfg.get("log_responses", True),
        max_retry=int(llm_cfg.get("max_retry", 3)),
    )
    # Build item_lookup once if dataset is available — used to send real names to LLM
    item_lookup = dataset.item_lookup() if dataset is not None else None
    return {
        "llm": llm,
        "translator": DirectionTranslator(llm, item_lookup=item_lookup),
        "hyp_generator": HypothesisGenerator(llm, item_lookup=item_lookup),
        "option_writer": OptionWriter(llm, include_none=cfg.get("ask", {}).get("include_none", True)),
    }


def run_interrec_experiment(
    cfg: dict[str, Any],
    run_dir: Path,
    dataset: InterRecDataset,
    index: EmbeddingIndex,
    logger: RunLogger,
    bm25_index: BM25Index | None = None,
) -> dict[str, Any]:
    sim_cfg = cfg.get("simulation", {})
    split = str(sim_cfg.get("split", "test"))
    max_users = int(sim_cfg.get("max_users", 50))
    top_k = int(cfg.get("recommendation", {}).get("top_k", 10))
    lambda_explore = float(cfg.get("recommendation", {}).get("lambda_explore", 0.1))

    components = build_components(cfg, dataset=dataset)
    recommender = BeliefRecommender(index, lambda_explore=lambda_explore)

    sessions = dataset.sessions[:max_users]

    all_results: list[dict[str, Any]] = []
    for session in tqdm(sessions, desc="Running InterRec episodes"):
        try:
            theta_info = build_theta_star(session, index, split=split, dataset=dataset)
            episode = run_session(
                session=session,
                theta_info=theta_info,
                index=index,
                recommender=recommender,
                translator=components["translator"],
                hyp_generator=components["hyp_generator"],
                option_writer=components["option_writer"],
                dataset=dataset,
                cfg=cfg,
                bm25_index=bm25_index,
            )
            all_results.append(episode)
            for turn_log in episode["turns"]:
                turn_log["run_id"] = logger.run_id
                turn_log["dataset"] = cfg.get("dataset", {}).get("name", "unknown")
                turn_log["method"] = "interrec"
                turn_log["seed"] = cfg.get("seed", 42)
                logger.log_turn(turn_log)
        except Exception:
            logger.log_error(session["user_id"], traceback.format_exc())

    aggregate = _aggregate_metrics(all_results, top_k)
    logger.write_metrics(aggregate, all_results)
    return aggregate


def run_bm25_baseline(
    cfg: dict[str, Any],
    dataset: InterRecDataset,
    index: EmbeddingIndex,
    logger: RunLogger,
    bm25_index: BM25Index,
) -> dict[str, Any]:
    """Static BM25 retrieval baseline: no belief update, no interaction."""
    from src.recommendation.metrics import ranking_metrics

    sim_cfg = cfg.get("simulation", {})
    max_users = int(sim_cfg.get("max_users", 50))
    split = str(sim_cfg.get("split", "test"))
    split_key = {"train": "future_train", "valid": "future_valid", "test": "future_test"}.get(split, "future_test")
    top_k = int(cfg.get("recommendation", {}).get("top_k", 10))

    sessions = dataset.sessions[:max_users]
    all_hr, all_ndcg, all_mrr = [], [], []

    for session in tqdm(sessions, desc="BM25 baseline"):
        observed = session["observed_history"]
        query = " ".join(str(x) for x in observed[-20:])
        relevant = set(str(x) for x in session.get(split_key, []))
        seen = set(str(x) for x in observed)
        results = bm25_index.search(query, top_k=top_k, exclude=seen)
        rec_ids = [r[0] for r in results]
        m = ranking_metrics(rec_ids, relevant, k=top_k)
        all_hr.append(m[f"HitRate@{top_k}"])
        all_ndcg.append(m[f"NDCG@{top_k}"])
        all_mrr.append(m[f"MRR@{top_k}"])

    aggregate = {
        "method": "bm25_baseline",
        "dataset": cfg.get("dataset", {}).get("name", "unknown"),
        "seed": cfg.get("seed", 42),
        f"HitRate@{top_k}": float(sum(all_hr) / max(len(all_hr), 1)),
        f"NDCG@{top_k}": float(sum(all_ndcg) / max(len(all_ndcg), 1)),
        f"MRR@{top_k}": float(sum(all_mrr) / max(len(all_mrr), 1)),
        "n_users": len(all_hr),
    }
    logger.write_metrics(aggregate, [])
    return aggregate


def _aggregate_metrics(results: list[dict[str, Any]], top_k: int = 10) -> dict[str, Any]:
    hr_list, ndcg_list, mrr_list, ask_list, pref_err_list = [], [], [], [], []
    for ep in results:
        turns = ep.get("turns", [])
        if not turns:
            continue
        last_m = turns[-1].get("metrics", {})
        hr_list.append(last_m.get(f"HitRate@{top_k}", 0.0))
        ndcg_list.append(last_m.get(f"NDCG@{top_k}", 0.0))
        mrr_list.append(last_m.get(f"MRR@{top_k}", 0.0))
        pref_err_list.append(last_m.get("preference_error", 0.0))
        ask_list.append(ep.get("total_ask_count", 0))

    def safe_mean(lst: list[float]) -> float:
        return float(sum(lst) / max(len(lst), 1))

    return {
        "method": "interrec",
        f"HitRate@{top_k}": safe_mean(hr_list),
        f"NDCG@{top_k}": safe_mean(ndcg_list),
        f"MRR@{top_k}": safe_mean(mrr_list),
        "avg_preference_error": safe_mean(pref_err_list),
        "avg_ask_count": safe_mean(ask_list),
        "n_users": len(results),
    }
