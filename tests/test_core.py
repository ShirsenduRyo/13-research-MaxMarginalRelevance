"""Unit tests for the core reranking utilities."""

import numpy as np
import torch

from rerank_paper.core import (
    aggregate_metrics,
    alpha_ndcg_at_k,
    build_budget_supervision,
    err_ia_at_k,
    evaluate_rankings,
    extract_query_features,
    intent_coverage_at_k,
    make_synthetic_encoded_queries,
    mmr_ranking,
    ndcg_any_intent_at_k,
    rank_by_relevance,
    rank_original_serp,
    set_reproducibility,
)


def test_mmr_returns_unique_valid_indices():
    item = next(iter(make_synthetic_encoded_queries(n_queries=1).values()))
    order = mmr_ranking(item, lambda_mult=0.65, top_k=5)
    assert len(order) == 5
    assert len(set(order)) == 5
    assert all(0 <= index < len(item.record.candidate_ids) for index in order)


def test_mmr_lambda_one_matches_relevance_order():
    item = next(iter(make_synthetic_encoded_queries(n_queries=1).values()))
    assert mmr_ranking(item, lambda_mult=1.0, top_k=10) == rank_by_relevance(item)


def test_original_serp_order_is_identity():
    item = next(iter(make_synthetic_encoded_queries(n_queries=1).values()))
    assert rank_original_serp(item) == list(range(len(item.record.candidate_ids)))


def test_alpha_ndcg_and_coverage_are_bounded():
    qrels = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    ranking = [0, 1, 2]
    score = alpha_ndcg_at_k(ranking, qrels, k=3, alpha=0.5)
    assert np.isclose(score, 1.0)
    assert np.isclose(intent_coverage_at_k(ranking, qrels, k=2), 1.0)
    assert 0.0 <= ndcg_any_intent_at_k(ranking, qrels, k=3) <= 1.0


def test_err_ia_binary_example():
    qrels = np.array([[1, 0], [0, 1]], dtype=np.float32)
    assert np.isclose(err_ia_at_k([0, 1], qrels, k=2), 0.75)


def test_evaluation_and_aggregation_shape():
    encoded = make_synthetic_encoded_queries(n_queries=5, n_candidates=10, seed=5)
    rankers = {
        "serp": rank_original_serp,
        "mmr": lambda item: mmr_ranking(item, lambda_mult=0.65, top_k=10),
    }
    per_query = evaluate_rankings(encoded, rankers, k_values=(5, 10))
    summary = aggregate_metrics(per_query)
    assert len(per_query) == 5 * 2 * 2
    assert {"mean", "metric", "method", "k"}.issubset(summary.columns)


def test_budget_supervision_has_features_and_valid_actions():
    encoded = make_synthetic_encoded_queries(n_queries=12, n_candidates=10, seed=11)
    labels = build_budget_supervision(
        encoded_queries=encoded,
        action_depths=(6, 8, 10),
        top_k=5,
        lambda_mult=0.65,
        feasible_depths={6, 8},
        alpha=0.5,
    )
    assert len(labels) == 12
    assert set(labels["target_depth"]).issubset({6, 8})
    features = extract_query_features(next(iter(encoded.values())))
    assert "n_candidates" in features
    assert all(np.isfinite(list(features.values())))


def test_seed_helper_and_cpu_embeddings():
    set_reproducibility(123)
    encoded = make_synthetic_encoded_queries(n_queries=1, seed=123)
    item = next(iter(encoded.values()))
    assert item.device.type == "cpu"
    assert torch.isfinite(item.query_embedding).all()
