"""Core utilities for reproducible diversity-aware reranking experiments.

The module intentionally keeps ranking, data handling, metrics, and timing in
small functions so experiments can be audited and changed independently.
"""

from __future__ import annotations

import json
import math
import pickle
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class QueryRecord:
    """One query and its candidate documents with intent-level judgments.

    Attributes:
        qid: Stable query ID from the benchmark.
        query: The original query text.
        candidate_ids: Candidate IDs in first-stage/SERP order.
        candidate_texts: Candidate title/snippet text in the same order.
        intents: Intent labels available for evaluation.
        intent_relevance: Binary matrix of shape (n_intents, n_candidates).
    """

    qid: str
    query: str
    candidate_ids: tuple[str, ...]
    candidate_texts: tuple[str, ...]
    intents: tuple[str, ...]
    intent_relevance: np.ndarray

    def __post_init__(self) -> None:
        if len(self.candidate_ids) != len(self.candidate_texts):
            raise ValueError("candidate_ids and candidate_texts must have equal length")
        qrels = np.asarray(self.intent_relevance)
        if qrels.ndim != 2:
            raise ValueError("intent_relevance must be a 2D (intent, document) matrix")
        if qrels.shape != (len(self.intents), len(self.candidate_ids)):
            raise ValueError(
                "intent_relevance shape must match (len(intents), len(candidate_ids)); "
                f"got {qrels.shape}"
            )
        if not np.isfinite(qrels).all():
            raise ValueError("intent_relevance contains non-finite values")


@dataclass
class EncodedQuery:
    """A query record with normalized query/document embeddings on one device."""

    record: QueryRecord
    query_embedding: torch.Tensor
    document_embeddings: torch.Tensor

    def __post_init__(self) -> None:
        if self.query_embedding.ndim != 1:
            raise ValueError("query_embedding must be a 1D tensor")
        if self.document_embeddings.ndim != 2:
            raise ValueError("document_embeddings must be a 2D tensor")
        if self.document_embeddings.shape[0] != len(self.record.candidate_ids):
            raise ValueError("Number of document embeddings does not match candidate count")
        if self.document_embeddings.shape[1] != self.query_embedding.shape[0]:
            raise ValueError("Query/document embedding dimensions do not match")

    @property
    def relevance_scores(self) -> torch.Tensor:
        """Cosine relevance scores (embeddings are normalized)."""
        return self.document_embeddings @ self.query_embedding

    @property
    def device(self) -> torch.device:
        return self.query_embedding.device


def set_reproducibility(seed: int = 42, torch_num_threads: int = 1) -> None:
    """Seed Python/NumPy/PyTorch and set CPU threads for stable microbenchmarks."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch_num_threads > 0:
        torch.set_num_threads(torch_num_threads)


def choose_device(requested: str = "auto") -> str:
    """Resolve auto/cpu/cuda into a device string; fail clearly if CUDA is absent."""
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return requested


def load_mimics_records(
    data_dir: str | Path,
    split: str = "all",
    max_queries: int | None = None,
    min_candidates: int = 1,
) -> list[QueryRecord]:
    """Load the processed MIMICS-Diversification files into stable query records.

    Expected files (as documented by the upstream repository):
      serps.pkl, id2doc.pkl, query2qid.json, judgement.tsv; and, for split
      train/test, train_qids.npy or test_qids.npy.

    The official processing script creates keyword-matched intent labels. Treat
    these labels as weak supervision/proxies, not manual judgments.
    """
    root = Path(data_dir).expanduser().resolve()
    required = ["serps.pkl", "id2doc.pkl", "query2qid.json", "judgement.tsv"]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing processed MIMICS files in {root}: {missing}. "
            "See the upstream MIMICS-Diversification README for source-data preparation."
        )
    if split not in {"all", "train", "test"}:
        raise ValueError("split must be one of: 'all', 'train', 'test'")

    with (root / "serps.pkl").open("rb") as file:
        serps: dict[str, list[Any]] = pickle.load(file)
    with (root / "id2doc.pkl").open("rb") as file:
        id2doc: dict[Any, str] = pickle.load(file)
    with (root / "query2qid.json").open("r", encoding="utf-8") as file:
        query2qid: dict[str, Any] = json.load(file)

    judgement_path = root / "judgement.tsv"
    judgements = pd.read_csv(judgement_path, sep="\t", dtype=str)
    required_columns = {"query", "intent", "doc_id", "judgement"}
    normalized_columns = {str(column).strip().lower(): column for column in judgements.columns}
    if not required_columns.issubset(normalized_columns):
        # Some TSV exports omit their header. Retry using the documented schema.
        judgements = pd.read_csv(
            judgement_path,
            sep="\t",
            dtype=str,
            header=None,
            names=["query", "intent", "doc_id", "judgement"],
        )
        first_value = str(judgements.iloc[0]["query"]).strip().lower() if len(judgements) else ""
        if first_value == "query":
            judgements = judgements.iloc[1:].reset_index(drop=True)
        normalized_columns = {str(column).strip().lower(): column for column in judgements.columns}

    judgements = judgements.rename(columns={
        normalized_columns[name]: name for name in required_columns if name in normalized_columns
    })
    if not required_columns.issubset(judgements.columns):
        raise ValueError(
            f"judgement.tsv must contain columns {sorted(required_columns)}; "
            f"found {list(judgements.columns)}"
        )
    judgements = judgements[list(sorted(required_columns))].copy()
    for column in ["query", "intent", "doc_id"]:
        judgements[column] = judgements[column].fillna("").astype(str)
    judgements["judgement"] = pd.to_numeric(
        judgements["judgement"], errors="coerce"
    ).fillna(0.0)
    judgements["relevant"] = (judgements["judgement"] > 0).astype(np.float32)

    allowed_qids: set[str] | None = None
    if split != "all":
        split_path = root / f"{split}_qids.npy"
        if not split_path.is_file():
            raise FileNotFoundError(f"Expected split file not found: {split_path}")
        allowed_qids = {str(value) for value in np.load(split_path, allow_pickle=False).tolist()}

    judgments_by_query = {
        str(query): group.copy()
        for query, group in judgements.groupby("query", sort=False)
    }
    query_items = sorted(
        serps.items(),
        key=lambda item: (str(query2qid.get(str(item[0]), "~")), str(item[0])),
    )
    records: list[QueryRecord] = []
    for query, raw_candidate_ids in query_items:
        query_text = str(query)
        qid = str(query2qid.get(query_text, ""))
        if allowed_qids is not None and qid not in allowed_qids:
            continue
        if query_text not in judgments_by_query:
            continue

        candidate_ids: list[str] = []
        candidate_texts: list[str] = []
        seen_ids: set[str] = set()
        for raw_doc_id in raw_candidate_ids:
            doc_id = str(raw_doc_id)
            # IDs loaded from pickle are commonly strings; allow exact string match.
            resolved_id: Any = raw_doc_id if raw_doc_id in id2doc else doc_id
            if resolved_id not in id2doc or doc_id in seen_ids:
                continue
            candidate_ids.append(doc_id)
            candidate_texts.append(str(id2doc[resolved_id] or ""))
            seen_ids.add(doc_id)
        if len(candidate_ids) < min_candidates:
            continue

        query_judgements = judgments_by_query[query_text]
        candidate_index = {doc_id: i for i, doc_id in enumerate(candidate_ids)}
        intents = tuple(dict.fromkeys(query_judgements["intent"].tolist()))
        intent_index = {intent: i for i, intent in enumerate(intents)}
        qrels = np.zeros((len(intents), len(candidate_ids)), dtype=np.float32)
        for row in query_judgements.itertuples(index=False):
            doc_id = str(getattr(row, "doc_id"))
            intent = str(getattr(row, "intent"))
            if doc_id in candidate_index and intent in intent_index:
                i = intent_index[intent]
                j = candidate_index[doc_id]
                qrels[i, j] = max(qrels[i, j], float(getattr(row, "relevant")))

        if not intents or not candidate_ids:
            continue
        records.append(QueryRecord(
            qid=qid or query_text,
            query=query_text,
            candidate_ids=tuple(candidate_ids),
            candidate_texts=tuple(candidate_texts),
            intents=intents,
            intent_relevance=qrels,
        ))
        if max_queries is not None and len(records) >= max_queries:
            break

    if not records:
        raise ValueError(
            f"No query records loaded from {root}. Check split files, query IDs and judgments."
        )
    return records


def encode_records(
    records: Sequence[QueryRecord],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    device: str = "auto",
    show_progress_bar: bool = True,
) -> dict[str, EncodedQuery]:
    """Encode queries and unique candidate documents with SentenceTransformers.

    Returned vectors are kept on CPU so the encoded cache does not hold GPU VRAM.
    Move them to a target device with ``to_device`` for an explicit timing run.
    """
    if not records:
        raise ValueError("records must contain at least one query")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Install sentence-transformers before encoding: pip install sentence-transformers"
        ) from exc

    model_device = choose_device(device)
    model = SentenceTransformer(model_name, device=model_device)
    doc_text_by_id: dict[str, str] = {}
    for record in records:
        for doc_id, text in zip(record.candidate_ids, record.candidate_texts):
            doc_text_by_id.setdefault(doc_id, text)

    unique_doc_ids = list(doc_text_by_id)
    unique_doc_texts = [doc_text_by_id[doc_id] for doc_id in unique_doc_ids]
    document_vectors = model.encode(
        unique_doc_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=show_progress_bar,
    )
    query_vectors = model.encode(
        [record.query for record in records],
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=show_progress_bar,
    )

    doc_index = {doc_id: i for i, doc_id in enumerate(unique_doc_ids)}
    document_tensor = torch.as_tensor(np.asarray(document_vectors, dtype=np.float32))
    query_tensor = torch.as_tensor(np.asarray(query_vectors, dtype=np.float32))
    encoded: dict[str, EncodedQuery] = {}
    for record_index, record in enumerate(records):
        indices = [doc_index[doc_id] for doc_id in record.candidate_ids]
        encoded[record.qid] = EncodedQuery(
            record=record,
            query_embedding=query_tensor[record_index].detach().cpu().contiguous(),
            document_embeddings=document_tensor[indices].detach().cpu().contiguous(),
        )
    return encoded


def to_device(
    encoded: Mapping[str, EncodedQuery], device: str | torch.device
) -> dict[str, EncodedQuery]:
    """Create encoded-query views on the requested device for ranking/benchmarking."""
    target = torch.device(device)
    return {
        qid: EncodedQuery(
            record=item.record,
            query_embedding=item.query_embedding.to(target),
            document_embeddings=item.document_embeddings.to(target),
        )
        for qid, item in encoded.items()
    }


def rank_original_serp(encoded: EncodedQuery, top_k: int | None = None) -> list[int]:
    """Return candidate indices in original SERP order."""
    n_docs = len(encoded.record.candidate_ids)
    result = list(range(n_docs))
    return result if top_k is None else result[:max(0, min(top_k, n_docs))]


def rank_by_relevance(encoded: EncodedQuery, top_k: int | None = None) -> list[int]:
    """Rank all candidates by query/document cosine similarity, highest first."""
    scores = encoded.relevance_scores.detach().cpu().numpy()
    # Stable sorting makes tie handling reproducible.
    result = np.argsort(-scores, kind="stable").tolist()
    return result if top_k is None else result[:max(0, min(top_k, len(result)))]


def mmr_ranking(
    encoded: EncodedQuery,
    lambda_mult: float = 0.65,
    top_k: int = 10,
    candidate_depth: int | None = None,
    redundancy_floor: float | None = 0.0,
) -> list[int]:
    """Greedy MMR with incremental maximum-redundancy updates.

    Candidate depth, when supplied, first keeps the top-M candidates by query
    relevance. The result indices always refer to the original record candidate
    order. ``redundancy_floor=0.0`` avoids rewarding negative cosine similarity;
    set it to None to use raw cosine values as the redundancy signal.
    """
    if not 0.0 <= lambda_mult <= 1.0:
        raise ValueError("lambda_mult must be between 0 and 1 inclusive")
    if top_k < 1:
        return []

    n_docs = encoded.document_embeddings.shape[0]
    if n_docs == 0:
        return []
    relevance = encoded.relevance_scores
    if candidate_depth is not None:
        if candidate_depth < 1:
            return []
        candidate_indices = torch.argsort(relevance, descending=True, stable=True)[:candidate_depth]
    else:
        candidate_indices = torch.arange(n_docs, device=relevance.device)

    candidate_vectors = encoded.document_embeddings.index_select(0, candidate_indices)
    candidate_relevance = relevance.index_select(0, candidate_indices)
    n_candidates = candidate_vectors.shape[0]
    n_select = min(top_k, n_candidates)
    available = torch.ones(n_candidates, dtype=torch.bool, device=candidate_vectors.device)
    max_redundancy = torch.zeros(n_candidates, dtype=candidate_vectors.dtype,
                                  device=candidate_vectors.device)
    selected_local: list[int] = []

    for _ in range(n_select):
        scores = lambda_mult * candidate_relevance - (1.0 - lambda_mult) * max_redundancy
        scores = scores.masked_fill(~available, -torch.inf)
        chosen = int(torch.argmax(scores).item())
        selected_local.append(chosen)
        available[chosen] = False
        similarities = candidate_vectors @ candidate_vectors[chosen]
        if redundancy_floor is not None:
            similarities = torch.clamp(similarities, min=redundancy_floor)
        max_redundancy = torch.maximum(max_redundancy, similarities)

    original_indices = candidate_indices.detach().cpu().numpy().astype(int).tolist()
    return [original_indices[index] for index in selected_local]


def _alpha_dcg(order: Sequence[int], qrels: np.ndarray, alpha: float, k: int) -> float:
    if qrels.size == 0:
        return 0.0
    gains_seen = np.zeros(qrels.shape[0], dtype=np.float64)
    dcg = 0.0
    for rank, doc_index in enumerate(list(order)[:k], start=1):
        gains = (qrels[:, int(doc_index)] > 0).astype(np.float64)
        marginal_gain = np.sum(gains * np.power(1.0 - alpha, gains_seen))
        dcg += float(marginal_gain / math.log2(rank + 1.0))
        gains_seen += gains
    return dcg


def _ideal_alpha_dcg(qrels: np.ndarray, alpha: float, k: int) -> float:
    if qrels.size == 0:
        return 0.0
    remaining = list(range(qrels.shape[1]))
    selected: list[int] = []
    seen = np.zeros(qrels.shape[0], dtype=np.float64)
    for rank in range(1, min(k, qrels.shape[1]) + 1):
        if not remaining:
            break
        best_doc = None
        best_gain = -1.0
        for doc_index in remaining:
            gains = (qrels[:, doc_index] > 0).astype(np.float64)
            gain = float(np.sum(gains * np.power(1.0 - alpha, seen)))
            if gain > best_gain:
                best_gain = gain
                best_doc = doc_index
        if best_doc is None:
            break
        selected.append(best_doc)
        seen += (qrels[:, best_doc] > 0).astype(np.float64)
        remaining.remove(best_doc)
    return _alpha_dcg(selected, qrels, alpha=alpha, k=k)


def alpha_ndcg_at_k(
    order: Sequence[int], qrels: np.ndarray, k: int = 5, alpha: float = 0.5
) -> float:
    """Alpha-nDCG@k for binary intent/document judgments."""
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must satisfy 0 <= alpha < 1")
    qrels = np.asarray(qrels, dtype=np.float32)
    ideal = _ideal_alpha_dcg(qrels, alpha=alpha, k=k)
    if ideal <= 0.0:
        return 0.0
    return float(_alpha_dcg(order, qrels, alpha=alpha, k=k) / ideal)


def err_ia_at_k(order: Sequence[int], qrels: np.ndarray, k: int = 5) -> float:
    """ERR-IA@k under uniform intent priors and binary satisfaction judgments."""
    qrels = (np.asarray(qrels) > 0).astype(np.float64)
    if qrels.ndim != 2 or qrels.shape[0] == 0:
        return 0.0
    per_intent: list[float] = []
    for intent_index in range(qrels.shape[0]):
        continuation = 1.0
        err = 0.0
        for rank, doc_index in enumerate(list(order)[:k], start=1):
            if doc_index < 0 or doc_index >= qrels.shape[1]:
                continue
            satisfaction = float(qrels[intent_index, int(doc_index)])
            err += continuation * satisfaction / rank
            continuation *= (1.0 - satisfaction)
            if continuation == 0.0:
                break
        per_intent.append(err)
    return float(np.mean(per_intent)) if per_intent else 0.0


def intent_coverage_at_k(order: Sequence[int], qrels: np.ndarray, k: int = 5) -> float:
    """Fraction of judged-reachable intents represented at least once in top-k."""
    qrels = (np.asarray(qrels) > 0).astype(np.int8)
    if qrels.ndim != 2 or qrels.shape[0] == 0:
        return 0.0
    reachable = qrels.sum(axis=1) > 0
    denominator = int(reachable.sum())
    if denominator == 0:
        return 0.0
    selected = [int(i) for i in list(order)[:k] if 0 <= int(i) < qrels.shape[1]]
    if not selected:
        return 0.0
    covered = (qrels[:, selected].sum(axis=1) > 0) & reachable
    return float(covered.sum() / denominator)


def ndcg_any_intent_at_k(order: Sequence[int], qrels: np.ndarray, k: int = 5) -> float:
    """Binary nDCG proxy: relevant if judged relevant to any intent.

    This is a coarse relevance-preservation measure for MIMICS, not a replacement
    for standard single-intent qrels on a conventional ad-hoc IR benchmark.
    """
    qrels = (np.asarray(qrels) > 0).astype(np.float64)
    if qrels.ndim != 2 or qrels.shape[1] == 0:
        return 0.0
    labels = qrels.max(axis=0)
    selected = [int(i) for i in list(order)[:k] if 0 <= int(i) < len(labels)]
    dcg = sum(float(labels[index]) / math.log2(rank + 1.0)
              for rank, index in enumerate(selected, start=1))
    ideal_labels = np.sort(labels)[::-1][:k]
    ideal_dcg = sum(float(label) / math.log2(rank + 1.0)
                    for rank, label in enumerate(ideal_labels, start=1))
    return float(dcg / ideal_dcg) if ideal_dcg > 0 else 0.0


def mean_pairwise_cosine_at_k(
    order: Sequence[int], document_embeddings: torch.Tensor, k: int = 5
) -> float:
    """Mean pairwise cosine similarity among selected items; lower suggests less redundancy."""
    selected = [int(i) for i in list(order)[:k]
                if 0 <= int(i) < document_embeddings.shape[0]]
    if len(selected) < 2:
        return float("nan")
    vectors = F.normalize(document_embeddings[selected].detach(), p=2, dim=1)
    similarities = vectors @ vectors.T
    upper = torch.triu_indices(len(selected), len(selected), offset=1,
                               device=similarities.device)
    return float(similarities[upper[0], upper[1]].mean().detach().cpu().item())


def evaluate_rankings(
    encoded: Mapping[str, EncodedQuery],
    rankers: Mapping[str, Callable[[EncodedQuery], Sequence[int]]],
    k_values: Sequence[int] = (5, 10),
    alpha: float = 0.5,
) -> pd.DataFrame:
    """Evaluate rankers query-by-query and return a long-form pandas DataFrame."""
    rows: list[dict[str, Any]] = []
    for qid, item in encoded.items():
        for method_name, ranker in rankers.items():
            order = list(ranker(item))
            for k in k_values:
                row: dict[str, Any] = {
                    "qid": qid,
                    "method": method_name,
                    "k": int(k),
                    "n_candidates": len(item.record.candidate_ids),
                    "alpha_ndcg": alpha_ndcg_at_k(order, item.record.intent_relevance, k=k, alpha=alpha),
                    "err_ia": err_ia_at_k(order, item.record.intent_relevance, k=k),
                    "intent_coverage": intent_coverage_at_k(order, item.record.intent_relevance, k=k),
                    "ndcg_any_intent": ndcg_any_intent_at_k(order, item.record.intent_relevance, k=k),
                }
                row["mean_pairwise_cosine"] = mean_pairwise_cosine_at_k(
                    order, item.document_embeddings, k=k
                )
                rows.append(row)
    return pd.DataFrame(rows)


def aggregate_metrics(
    per_query: pd.DataFrame,
    metrics: Sequence[str] = (
        "alpha_ndcg", "err_ia", "intent_coverage", "ndcg_any_intent", "mean_pairwise_cosine"
    ),
) -> pd.DataFrame:
    """Aggregate query-level metrics with mean, standard error and 95% normal CI."""
    group_columns = [column for column in ["method", "k"] if column in per_query.columns]
    rows: list[dict[str, Any]] = []
    for keys, group in per_query.groupby(group_columns, sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_columns, keys))
        for metric in metrics:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            if len(values) == 0:
                mean, stderr, low, high = np.nan, np.nan, np.nan, np.nan
            else:
                mean = float(values.mean())
                stderr = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
                low, high = mean - 1.96 * stderr, mean + 1.96 * stderr
            rows.append({**base, "metric": metric, "n_queries": int(len(values)),
                         "mean": mean, "stderr": stderr, "ci95_low_normal": low,
                         "ci95_high_normal": high})
    return pd.DataFrame(rows)


def synchronize_device(device: torch.device) -> None:
    """Synchronize CUDA before/after timers because GPU execution is asynchronous."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_ranker(
    encoded_queries: Sequence[EncodedQuery],
    ranker: Callable[[EncodedQuery], Sequence[int]],
    warmup_calls: int = 5,
    repeats_per_query: int = 5,
    max_queries: int = 100,
) -> dict[str, Any]:
    """Microbenchmark a ranking callable; returns per-call latency distribution.

    The ranker must receive already encoded query/document vectors. This excludes
    query/document embedding time. The caller should separately report end-to-end
    embedding cost. Run all compared methods on the same machine and device.
    """
    if not encoded_queries:
        raise ValueError("encoded_queries cannot be empty")
    if warmup_calls < 0 or repeats_per_query < 1 or max_queries < 1:
        raise ValueError("Invalid benchmark repeat/query configuration")
    sample = list(encoded_queries[:max_queries])
    device = sample[0].device
    for item in sample:
        if item.device != device:
            raise ValueError("All encoded queries in one benchmark must be on the same device")

    for item in sample[:min(len(sample), warmup_calls)]:
        ranker(item)
    synchronize_device(device)

    latencies_ms: list[float] = []
    for item in sample:
        for _ in range(repeats_per_query):
            synchronize_device(device)
            start = time.perf_counter_ns()
            ranker(item)
            synchronize_device(device)
            elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
            latencies_ms.append(elapsed_ms)

    values = np.asarray(latencies_ms, dtype=np.float64)
    total_seconds = float(values.sum() / 1000.0)
    return {
        "n_queries": len(sample),
        "n_calls": len(values),
        "device": str(device),
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
        "calls_per_second_serial": float(len(values) / total_seconds) if total_seconds > 0 else math.inf,
        "latencies_ms": values,
    }


def extract_query_features(encoded: EncodedQuery, max_docs_for_similarity: int = 10) -> dict[str, float]:
    """Extract inexpensive inference-time features using NumPy on CPU.

    No intent judgments or test labels are used. Only the highest-relevance
    candidates are used for the small pairwise-similarity summary. This avoids
    repeated tiny PyTorch kernel dispatches during CPU serving; moving embeddings
    from CUDA to CPU here would add transfer overhead, so profile the controller
    on the device where it is meant to run.
    """
    query_vector = encoded.query_embedding.detach().to("cpu").numpy().astype(np.float32, copy=False)
    document_vectors = encoded.document_embeddings.detach().to("cpu").numpy().astype(np.float32, copy=False)
    scores = document_vectors @ query_vector
    n_docs = len(scores)
    sorted_scores = np.sort(scores)[::-1]
    top = sorted_scores[:min(5, n_docs)]
    top_m = min(max_docs_for_similarity, n_docs)
    candidate_order = np.argsort(-scores, kind="stable")[:top_m]
    vectors = document_vectors[candidate_order]
    vector_norms = np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-12)
    vectors = vectors / vector_norms
    sims = vectors @ vectors.T
    if top_m > 1:
        pairwise = sims[np.triu_indices(top_m, k=1)]
        mean_pairwise = float(np.mean(pairwise))
        max_pairwise = float(np.max(pairwise))
    else:
        mean_pairwise, max_pairwise = 0.0, 0.0
    sim_top_to_rest = sims[0, 1:] if top_m > 1 else np.asarray([0.0])
    shifted = scores - np.max(scores) if n_docs else scores
    probs = np.exp(shifted)
    probs = probs / max(float(probs.sum()), 1e-12)
    entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
    return {
        "n_candidates": float(n_docs),
        "relevance_top1": float(sorted_scores[0]) if n_docs else 0.0,
        "relevance_top2": float(sorted_scores[1]) if n_docs > 1 else 0.0,
        "relevance_top1_top2_gap": float(sorted_scores[0] - sorted_scores[1]) if n_docs > 1 else 0.0,
        "relevance_mean": float(np.mean(scores)) if n_docs else 0.0,
        "relevance_std": float(np.std(scores)) if n_docs else 0.0,
        "relevance_top5_mean": float(np.mean(top)) if len(top) else 0.0,
        "relevance_softmax_entropy": entropy,
        "mean_pairwise_cosine_top_candidates": mean_pairwise,
        "max_pairwise_cosine_top_candidates": max_pairwise,
        "mean_similarity_to_top_candidate": float(np.mean(sim_top_to_rest)),
    }


def build_budget_supervision(
    encoded_queries: Mapping[str, EncodedQuery],
    action_depths: Sequence[int],
    top_k: int,
    lambda_mult: float,
    feasible_depths: set[int],
    alpha: float = 0.5,
) -> pd.DataFrame:
    """Create train-only action labels from observed intent judgments.

    For each training query, choose the feasible candidate depth that maximizes
    alpha-nDCG@k; ties prefer smaller candidate depth. Never call this on test data
    to train the controller. Test judgments are only for final evaluation.
    """
    depths = sorted({int(depth) for depth in action_depths})
    if not depths or top_k < 1:
        raise ValueError("action_depths and top_k must be valid")
    feasible = set(int(depth) for depth in feasible_depths)
    if not feasible:
        raise ValueError("At least one depth must be feasible under the budget")
    rows: list[dict[str, Any]] = []
    for qid, item in encoded_queries.items():
        features = extract_query_features(item)
        outcomes: list[tuple[float, int]] = []
        for requested_depth in depths:
            if requested_depth not in feasible:
                continue
            effective_depth = min(requested_depth, len(item.record.candidate_ids))
            order = mmr_ranking(item, lambda_mult=lambda_mult, top_k=top_k,
                                candidate_depth=effective_depth)
            score = alpha_ndcg_at_k(order, item.record.intent_relevance, k=top_k, alpha=alpha)
            outcomes.append((score, requested_depth))
        if not outcomes:
            continue
        # max score; for exact ties select the shallower requested candidate pool.
        best_score = max(score for score, _ in outcomes)
        best_depth = min(depth for score, depth in outcomes if np.isclose(score, best_score))
        rows.append({"qid": qid, **features, "target_depth": int(best_depth),
                     "best_train_alpha_ndcg": float(best_score)})
    return pd.DataFrame(rows)


def select_budget_labels(
    encoded_queries: Mapping[str, EncodedQuery],
    action_depths: Sequence[int],
    top_k: int,
    lambda_mult: float,
    latency_p95_by_depth: Mapping[int, float],
    budget_ms: float,
    alpha: float = 0.5,
) -> pd.DataFrame:
    """Create budget-feasible, query-specific oracle labels for training only."""
    feasible_depths = {
        int(depth) for depth, latency in latency_p95_by_depth.items()
        if float(latency) <= float(budget_ms)
    }
    if not feasible_depths:
        # Always retain the empirically fastest calibrated action as a fallback.
        fastest_depth = min(latency_p95_by_depth, key=latency_p95_by_depth.get)
        feasible_depths = {int(fastest_depth)}
    return build_budget_supervision(
        encoded_queries=encoded_queries,
        action_depths=action_depths,
        top_k=top_k,
        lambda_mult=lambda_mult,
        feasible_depths=feasible_depths,
        alpha=alpha,
    )


def make_synthetic_encoded_queries(
    n_queries: int = 30,
    n_candidates: int = 10,
    embedding_dim: int = 64,
    n_intents: int = 4,
    seed: int = 123,
) -> dict[str, EncodedQuery]:
    """Create deterministic toy examples for software smoke tests only.

    The generated values are not experimental evidence and must never appear as
    empirical results in a paper.
    """
    if min(n_queries, n_candidates, embedding_dim, n_intents) < 1:
        raise ValueError("All synthetic dimensions must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result: dict[str, EncodedQuery] = {}
    for q in range(n_queries):
        query_vector = torch.randn(embedding_dim, generator=generator)
        query_vector = F.normalize(query_vector, p=2, dim=0)
        document_vectors = torch.randn(n_candidates, embedding_dim, generator=generator)
        # Add a query-aligned component so relevance scores are non-degenerate.
        document_vectors = F.normalize(document_vectors + 0.35 * query_vector.unsqueeze(0), p=2, dim=1)
        qrels = (torch.rand((n_intents, n_candidates), generator=generator) > 0.72).numpy().astype(np.float32)
        for intent_idx in range(n_intents):
            if not qrels[intent_idx].any():
                qrels[intent_idx, intent_idx % n_candidates] = 1.0
        record = QueryRecord(
            qid=f"synthetic-{q:04d}",
            query=f"synthetic query {q}",
            candidate_ids=tuple(f"{q}-doc-{d}" for d in range(n_candidates)),
            candidate_texts=tuple(f"synthetic candidate {d} for query {q}" for d in range(n_candidates)),
            intents=tuple(f"intent-{i}" for i in range(n_intents)),
            intent_relevance=qrels,
        )
        result[record.qid] = EncodedQuery(record, query_vector, document_vectors)
    return result
