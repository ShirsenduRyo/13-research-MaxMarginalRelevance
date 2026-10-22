"""Research utilities for diversity-aware reranking experiments."""

from .core import (
    EncodedQuery,
    QueryRecord,
    aggregate_metrics,
    alpha_ndcg_at_k,
    benchmark_ranker,
    build_budget_supervision,
    encode_records,
    evaluate_rankings,
    err_ia_at_k,
    extract_query_features,
    intent_coverage_at_k,
    load_mimics_records,
    make_synthetic_encoded_queries,
    mmr_ranking,
    ndcg_any_intent_at_k,
    rank_by_relevance,
    rank_original_serp,
    select_budget_labels,
    set_reproducibility,
    to_device,
)

__all__ = [
    "EncodedQuery", "QueryRecord", "aggregate_metrics", "alpha_ndcg_at_k",
    "benchmark_ranker", "build_budget_supervision", "encode_records",
    "evaluate_rankings", "err_ia_at_k", "extract_query_features",
    "intent_coverage_at_k", "load_mimics_records", "make_synthetic_encoded_queries",
    "mmr_ranking", "ndcg_any_intent_at_k", "rank_by_relevance", "rank_original_serp",
    "select_budget_labels", "set_reproducibility", "to_device",
]
