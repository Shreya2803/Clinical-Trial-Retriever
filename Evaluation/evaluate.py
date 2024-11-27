import numpy as np
import sys
import re

def dcg(relevances, rank=10):
    """Discounted cumulative gain at rank (DCG)"""
    relevances = np.asarray(relevances)[:rank]
    n_relevances = len(relevances)
    if n_relevances == 0:
        return 0.

    discounts = np.log2(np.arange(n_relevances) + 2)
    return np.sum(relevances / discounts)

def ndcg(relevances, rank=10):
    """Normalized discounted cumulative gain (NDCG)"""
    best_dcg = dcg(sorted(relevances, reverse=True), rank)
    if best_dcg == 0:
        return 0.

    return dcg(relevances, rank) / best_dcg

def precision_at_k(relevances, k=10):
    """Precision at k (P@k)"""
    relevances = np.asarray(relevances)[:k]
    return np.sum(relevances) / k

def reciprocal_rank(relevances):
    """Reciprocal rank (RecipRank)"""
    for i, rel in enumerate(relevances):
        if rel > 0:
            return 1 / (i + 1)
    return 0

def load_qrels(qrels_file):
    """Load the qrels file and return a dictionary mapping query_id to relevance scores."""
    qrels = {}
    with open(qrels_file, 'r') as f:
        for line in f:
            
            if not line.strip():
                continue
            parts = line.strip().split()

            if len(parts) != 4:
                print(f"Skipping malformed line: {line.strip()}")
                continue

            query_id, _, doc_id, relevance = parts
            relevance = int(relevance)  
            if query_id not in qrels:
                qrels[query_id] = {}
            qrels[query_id][doc_id] = relevance
    return qrels

def load_retrieved(retrieved_file):
    """Load the retrieved file and return a dictionary mapping query_id to retrieved docs and scores."""
    retrieved = {}
    with open(retrieved_file, 'r') as f:
        for line in f:
            parts = line.strip().split()

            # Ensure that we unpack only the first, third, fourth, and fifth elements
            if len(parts) >= 5:
                query_id = parts[0]  # First element: query_id
                doc_id = parts[2]    # Third element: doc_id
                rank_str = parts[3]  # Fourth element: rank (may contain unexpected characters)
                score = parts[4]     # Fifth element: score

                rank_str = re.sub(r'\D', '', rank_str)  # Remove all non-digit characters
                try:
                    rank = int(rank_str)  # Convert rank to integer
                except ValueError:
                    print(f"Skipping line with invalid rank: {line.strip()}")
                    continue
                

                try:
                    score = float(score)  # Convert score to float
                except ValueError:
                    print(f"Skipping line with invalid score: {line.strip()}")
                    continue

                # Add the doc_id, rank, and score for the given query_id
                if query_id not in retrieved:
                    retrieved[query_id] = []
                retrieved[query_id].append((doc_id, rank, score))

    return retrieved

def evaluate(qrels_file, retrieved_file, output_file):
    """Evaluate the results by calculating NDCG, P@10, and RecipRank, and write to output file."""
    qrels = load_qrels(qrels_file)
    retrieved = load_retrieved(retrieved_file)

    with open(output_file, 'a') as out_file:
        for query_id in qrels:
            # Get the relevance scores for the query
            relevances = [qrels[query_id].get(doc_id, 0) for doc_id, _, _ in retrieved.get(query_id, [])]

            # Calculate metrics
            ndcg_score = ndcg(relevances)
            p_at_10 = precision_at_k(relevances)
            recip_rank = reciprocal_rank(relevances)

            out_file.write(f"{query_id} {ndcg_score:.4f} {p_at_10:.4f} {recip_rank:.4f}\n")

if __name__ == "__main__":
    qrels_file = sys.argv[1]
    retrieved_file = sys.argv[2]
    output_file = sys.argv[3]

    evaluate(qrels_file, retrieved_file, output_file)

