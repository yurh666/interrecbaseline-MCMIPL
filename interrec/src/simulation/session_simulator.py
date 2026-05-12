from __future__ import annotations

from typing import Any

import numpy as np

from src.agents.direction_translator import DirectionTranslator
from src.agents.hypothesis_generator import HypothesisGenerator
from src.agents.option_writer import OptionWriter
from src.ask.anchor_selector import build_anchor_directions
from src.ask.ask_policy import should_ask
from src.ask.hypothesis_vectorizer import vectorize_hypotheses
from src.ask.question_selector import select_best_question
from src.belief.initialization import initialize_belief
from src.belief.laplace_update import laplace_choice_update
from src.embedding.index import EmbeddingIndex
from src.recommendation.metrics import ranking_metrics
from src.recommendation.recommender import BeliefRecommender
from src.simulation.user_simulator import UserSimulator


def run_session(
    session: dict[str, Any],
    theta_info: dict[str, Any],
    index: EmbeddingIndex,
    recommender: BeliefRecommender,
    translator: DirectionTranslator,
    hyp_generator: HypothesisGenerator,
    option_writer: OptionWriter,
    dataset: Any,
    cfg: dict[str, Any],
    bm25_index: Any = None,
) -> dict[str, Any]:
    """Run one complete episode (multiple turns) for a user session.

    Returns a list of per-turn log dicts.
    """
    ask_cfg = cfg.get("ask", {})
    sim_cfg = cfg.get("simulation", {})
    rec_cfg = cfg.get("recommendation", {})
    belief_cfg = cfg.get("belief", {})

    K_dir = int(ask_cfg.get("K_directions", 5))
    M_hyp = int(ask_cfg.get("M_hypotheses", 5))
    n_opt = int(ask_cfg.get("n_options", 3))
    anchor_k = int(ask_cfg.get("anchor_top_k", 5))
    rep_k = int(ask_cfg.get("representative_top_k", 5))
    tau = float(ask_cfg.get("tau_choice", 0.2))
    tau_none = float(ask_cfg.get("tau_none", 0.2))
    none_bias = float(ask_cfg.get("none_bias", 0.0))
    none_threshold = float(ask_cfg.get("none_threshold", 0.25))
    alpha_none = float(ask_cfg.get("alpha_none", 1.2))
    mc_samples = int(ask_cfg.get("mc_samples", 128))
    c_ask = float(ask_cfg.get("c_ask", 0.05))
    top_k = int(rec_cfg.get("top_k", 10))
    max_turns = int(sim_cfg.get("max_turns", 5))

    observed_history = session["observed_history"]
    split_key = theta_info["theta_star_source"]
    future_items = session.get("future_test", []) or session.get("future_valid", []) or session.get("future_train", [])
    relevant = set(str(x) for x in future_items)

    weights = dataset.weights_for_items(session["user_id"], observed_history)
    belief = initialize_belief(
        observed_history=observed_history,
        index=index,
        weights=weights,
        init_mode=belief_cfg.get("init_mode", "weighted_average"),
        sigma0=float(belief_cfg.get("sigma0", 1.0)),
        jitter=float(belief_cfg.get("jitter", 1e-6)),
    )

    theta_star = theta_info["theta_star"]
    simulator = UserSimulator(
        theta_star=theta_star,
        choice_mode=sim_cfg.get("user_choice_mode", "deterministic_argmax"),
        tau=float(sim_cfg.get("tau_user_choice", 0.2)),
        tau_none=tau_none,
        none_bias=none_bias,
        none_threshold=none_threshold,
    )

    seen_items = set(str(x) for x in observed_history)
    ask_count = 0
    turns_log: list[dict[str, Any]] = []

    impl_modes = {
        "item_encoder": cfg.get("embedding", {}).get("mode", "tfidf_svd"),
        "llm": translator.llm.implementation_mode,
        "direction_translator": translator.implementation_mode,
        "hypothesis_generator": hyp_generator.implementation_mode,
        "option_writer": option_writer.implementation_mode,
        "user_simulator": simulator.implementation_mode,
    }

    for turn in range(1, max_turns + 1):
        belief_before = belief.summary()

        # --- Recommend ---
        recs = recommender.recommend(belief, top_k=top_k, exclude=seen_items)
        rec_ids = [item_id for item_id, _ in recs]
        rec_scores = [score for _, score in recs]
        m5 = ranking_metrics(rec_ids, relevant, k=5)
        m10 = ranking_metrics(rec_ids, relevant, k=10)
        rec_log = {
            "recommended_items": rec_ids[:10],
            "scores": rec_scores[:10],
            **m5,
            **m10,
        }

        # --- Uncertainty directions + anchors ---
        anchor_dirs = build_anchor_directions(belief, index, K=K_dir, anchor_top_k=anchor_k, exclude=seen_items)
        anchor_dirs_log = [
            {
                "direction_id": d["direction_id"],
                "lambda": d["lambda"],
                "positive_anchors": d["positive_anchors"],
                "negative_anchors": d["negative_anchors"],
                "semantic_description": "",  # filled after translation
            }
            for d in anchor_dirs
        ]

        # --- Direction translation ---
        translated = translator.translate(anchor_dirs)
        for d_log, d_trans in zip(anchor_dirs_log, translated):
            d_log["semantic_description"] = d_trans.get("direction_name", "")

        # --- Hypothesis generation ---
        hypotheses = hyp_generator.generate(translated, observed_history, M=M_hyp)

        # --- Vectorize hypotheses ---
        # item_texts for keyword fallback (used when BM25 unavailable)
        item_texts = getattr(dataset, "_item_texts_cache", None)
        hyp_vectorized = vectorize_hypotheses(hypotheses, index, rep_k, bm25_index, item_texts=item_texts)
        hyp_log = [
            {
                "hypothesis_id": h["hypothesis_id"],
                "text_description": h["text_description"],
                "feature_signature": h.get("feature_signature", []),
                "representative_items": h.get("representative_items", []),
                "vector_norm": h.get("vector_norm", 0.0),
            }
            for h in hyp_vectorized
        ]

        # --- Question selection ---
        sel = select_best_question(
            belief, hyp_vectorized, n_opt, tau, none_bias, none_threshold, alpha_none, mc_samples
        )
        ig = sel["ig"]
        do_ask, voi = should_ask(ig, c_ask)

        q_sel_log = {
            "candidate_question_count": sel["candidate_count"],
            "best_question_ids": sel["best_ids"],
            "IG": ig,
            "VOI": voi,
            "ask": do_ask,
            "c_ask": c_ask,
        }

        # --- Question + user choice ---
        question_log: dict[str, Any] = {}
        user_choice_log: dict[str, Any] = {}
        belief_after_log: dict[str, Any] = {}

        if do_ask and sel["best_ids"]:
            ask_count += 1
            selected_hyps = [h for h in hyp_vectorized if h["hypothesis_id"] in sel["best_ids"]]
            rep_items_map = {h["hypothesis_id"]: h.get("representative_items", []) for h in selected_hyps}
            options = option_writer.write(selected_hyps, rep_items_map)
            option_ids = [o["option_id"] for o in options if o["option_id"] != "none"]
            option_texts = [o["option_text"] for o in options]
            opt_vectors = np.stack([h["vector"] for h in selected_hyps])

            choice_result = simulator.choose(opt_vectors, option_ids, include_none=True)
            selected_oid = choice_result["selected_option_id"]
            sel_vec_idx = choice_result["selected_vector_index"]

            updated_belief = laplace_choice_update(
                belief, opt_vectors, sel_vec_idx, tau, alpha_none
            )
            belief = updated_belief

            question_log = {
                "question_text": "Which of the following best describes your current interest?",
                "options": option_texts,
            }
            user_choice_log = {
                "theta_star_source": split_key,
                "theta_star_items": theta_info["theta_star_items"][:5],
                **choice_result,
            }
            belief_after_log = belief.summary()
        else:
            belief_after_log = belief_before.copy()

        # preference_error: ||mu_t - theta*||
        pref_error = float(np.linalg.norm(belief.mu - theta_star))

        turn_log: dict[str, Any] = {
            "user_id": session["user_id"],
            "episode_id": session.get("episode_id", session["user_id"]),
            "turn": turn,
            "data_split": {
                "observed_history_count": len(observed_history),
                "theta_star_source": split_key,
                "future_train_count": len(session.get("future_train", [])),
                "future_valid_count": len(session.get("future_valid", [])),
                "future_test_count": len(session.get("future_test", [])),
            },
            "implementation_modes": impl_modes,
            "belief_before": belief_before,
            "recommendation": rec_log,
            "uncertainty_directions": anchor_dirs_log,
            "hypotheses": hyp_log,
            "question_selection": q_sel_log,
            "question": question_log,
            "user_choice": user_choice_log,
            "belief_after": belief_after_log,
            "metrics": {
                "entropy_delta": belief_before["entropy"] - belief_after_log.get("entropy", belief_before["entropy"]),
                "preference_error": pref_error,
                **m10,
                "ask_count_so_far": ask_count,
            },
        }
        turns_log.append(turn_log)

    return {
        "user_id": session["user_id"],
        "episode_id": session.get("episode_id", session["user_id"]),
        "turns": turns_log,
        "total_ask_count": ask_count,
        "final_metrics": turns_log[-1]["metrics"] if turns_log else {},
    }
