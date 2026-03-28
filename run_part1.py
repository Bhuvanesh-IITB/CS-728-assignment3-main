import os
import json
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from utils import get_queries_and_items


# -------------------------
# Reproducibility
# -------------------------
def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------------
# Text preparation
# -------------------------
def simple_tokenize(text: str):
    """
    Lightweight tokenizer for BM25.
    """
    return text.lower().strip().split()


def build_tool_texts(tools: dict):
    """
    Build a consistent list of tool ids and tool texts.

    Each tool is represented using:
        tool_id + tool_description
    """
    tool_ids = list(tools.keys())
    tool_texts = [
        f"tool_id: {tool_id}. tool_description: {tools[tool_id]}"
        for tool_id in tool_ids
    ]
    return tool_ids, tool_texts


# -------------------------
# Metrics
# -------------------------
def compute_recall_metrics(all_ranked_tool_ids, test_queries):
    """
    all_ranked_tool_ids: list of ranked tool_id lists (one per query)
    test_queries: list of dicts with gold_tool_name

    Returns:
        recall@1, recall@5
    """
    total = len(test_queries)
    correct_at_1 = 0
    correct_at_5 = 0

    for ranked_tools, sample in zip(all_ranked_tool_ids, test_queries):
        gold = sample["gold_tool_name"]

        if len(ranked_tools) > 0 and ranked_tools[0] == gold:
            correct_at_1 += 1

        if gold in ranked_tools[:5]:
            correct_at_5 += 1

    recall_at_1 = correct_at_1 / total if total > 0 else 0.0
    recall_at_5 = correct_at_5 / total if total > 0 else 0.0

    return recall_at_1, recall_at_5


# -------------------------
# Save rankings
# -------------------------
def save_rankings(save_path, method_name, test_queries, ranked_tool_ids, ranked_scores):
    """
    Save ranked outputs for each query.
    """
    outputs = []

    for sample, tools_ranked, scores_ranked in zip(test_queries, ranked_tool_ids, ranked_scores):
        outputs.append({
            "qid": sample["qid"],
            "query": sample["text"],
            "gold_tool_name": sample["gold_tool_name"],
            "ranked_tools": tools_ranked,
            "ranked_scores": [float(x) for x in scores_ranked]
        })

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)

    print(f"[Saved] {method_name} rankings -> {save_path}")


# -------------------------
# BM25
# -------------------------
def run_bm25(test_queries, tool_ids, tool_texts):
    """
    Sparse retrieval baseline using BM25.
    """
    print("\n[BM25] Building index...")
    tokenized_corpus = [simple_tokenize(t) for t in tool_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    all_ranked_tool_ids = []
    all_ranked_scores = []

    print("[BM25] Ranking queries...")
    for sample in tqdm(test_queries):
        query = sample["text"]
        tokenized_query = simple_tokenize(query)

        scores = bm25.get_scores(tokenized_query)  # np array [num_tools]
        ranked_indices = np.argsort(scores)[::-1]

        ranked_tool_ids = [tool_ids[i] for i in ranked_indices]
        ranked_scores = [float(scores[i]) for i in ranked_indices]

        all_ranked_tool_ids.append(ranked_tool_ids)
        all_ranked_scores.append(ranked_scores)

    recall_at_1, recall_at_5 = compute_recall_metrics(all_ranked_tool_ids, test_queries)

    return {
        "method": "BM25",
        "recall@1": recall_at_1,
        "recall@5": recall_at_5,
        "ranked_tool_ids": all_ranked_tool_ids,
        "ranked_scores": all_ranked_scores,
    }


# -------------------------
# Dense retrieval helpers
# -------------------------
def mean_pooling(last_hidden_state, attention_mask):
    """
    Mean pooling for HF models if needed in future.
    Not used directly here because SentenceTransformer handles pooling.
    """
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def run_dense_retrieval(test_queries, tool_ids, tool_texts, model_name, method_name, device):
    """
    Dense retrieval:
        - encode tools independently
        - encode queries independently
        - cosine similarity
        - ranking
    """
    print(f"\n[{method_name}] Loading model: {model_name}")
    model = SentenceTransformer(model_name, device=device)

    print(f"[{method_name}] Encoding tools...")
    tool_embeddings = model.encode(
        tool_texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True
    )  # [num_tools, dim]

    all_ranked_tool_ids = []
    all_ranked_scores = []

    print(f"[{method_name}] Ranking queries...")
    for sample in tqdm(test_queries):
        query = sample["text"]

        query_embedding = model.encode(
            query,
            convert_to_tensor=True,
            normalize_embeddings=True
        )  # [dim]

        scores = torch.matmul(tool_embeddings, query_embedding)  # cosine similarity since normalized
        ranked_indices = torch.argsort(scores, descending=True)

        ranked_tool_ids = [tool_ids[i] for i in ranked_indices.tolist()]
        ranked_scores = [float(scores[i].item()) for i in ranked_indices]

        all_ranked_tool_ids.append(ranked_tool_ids)
        all_ranked_scores.append(ranked_scores)

    recall_at_1, recall_at_5 = compute_recall_metrics(all_ranked_tool_ids, test_queries)

    return {
        "method": method_name,
        "recall@1": recall_at_1,
        "recall@5": recall_at_5,
        "ranked_tool_ids": all_ranked_tool_ids,
        "ranked_scores": all_ranked_scores,
    }


# -------------------------
# Main
# -------------------------
def main(args):
    seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading dataset...")
    train_queries, test_queries, tools = get_queries_and_items()

    if args.max_test_queries is not None and args.max_test_queries > 0:
        test_queries = test_queries[:args.max_test_queries]
        print(f"[Debug] Using only first {len(test_queries)} test queries")

    tool_ids, tool_texts = build_tool_texts(tools)

    print(f"Number of test queries: {len(test_queries)}")
    print(f"Number of tools: {len(tool_ids)}")

    results_summary = []

    # -------------------------
    # 1) BM25
    # -------------------------
    bm25_results = run_bm25(test_queries, tool_ids, tool_texts)
    results_summary.append({
        "Method": bm25_results["method"],
        "Recall@1": bm25_results["recall@1"],
        "Recall@5": bm25_results["recall@5"],
    })

    save_rankings(
        save_path=os.path.join(args.output_dir, "bm25_rankings.json"),
        method_name="BM25",
        test_queries=test_queries,
        ranked_tool_ids=bm25_results["ranked_tool_ids"],
        ranked_scores=bm25_results["ranked_scores"],
    )

    # -------------------------
    # 2) msmarco-MiniLM
    # -------------------------
    minilm_results = run_dense_retrieval(
        test_queries=test_queries,
        tool_ids=tool_ids,
        tool_texts=tool_texts,
        model_name=args.minilm_model,
        method_name="msmarco-MiniLM",
        device=args.device,
    )
    results_summary.append({
        "Method": minilm_results["method"],
        "Recall@1": minilm_results["recall@1"],
        "Recall@5": minilm_results["recall@5"],
    })

    save_rankings(
        save_path=os.path.join(args.output_dir, "msmarco_minilm_rankings.json"),
        method_name="msmarco-MiniLM",
        test_queries=test_queries,
        ranked_tool_ids=minilm_results["ranked_tool_ids"],
        ranked_scores=minilm_results["ranked_scores"],
    )

    # -------------------------
    # 3) UAE-large-v1
    # -------------------------
    uae_results = run_dense_retrieval(
        test_queries=test_queries,
        tool_ids=tool_ids,
        tool_texts=tool_texts,
        model_name=args.uae_model,
        method_name="UAE-large-v1",
        device=args.device,
    )
    results_summary.append({
        "Method": uae_results["method"],
        "Recall@1": uae_results["recall@1"],
        "Recall@5": uae_results["recall@5"],
    })

    save_rankings(
        save_path=os.path.join(args.output_dir, "uae_large_v1_rankings.json"),
        method_name="UAE-large-v1",
        test_queries=test_queries,
        ranked_tool_ids=uae_results["ranked_tool_ids"],
        ranked_scores=uae_results["ranked_scores"],
    )

    # -------------------------
    # Save results table
    # -------------------------
    df_results = pd.DataFrame(results_summary)
    csv_path = os.path.join(args.output_dir, "part1_results.csv")
    df_results.to_csv(csv_path, index=False)

    print("\n==============================")
    print("Part 1 Results")
    print("==============================")
    print(df_results.to_string(index=False))
    print(f"\n[Saved] Results table -> {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="results_part1")
    parser.add_argument("--max_test_queries", type=int, default=None,
                        help="Use a smaller subset for debugging, e.g. 100")

    # Dense retrieval models
    parser.add_argument("--minilm_model", type=str, default="sentence-transformers/msmarco-MiniLM-L-6-v3")
    parser.add_argument("--uae_model", type=str, default="WhereIsAI/UAE-Large-V1")

    args = parser.parse_args()
    main(args)