from typing import List
from tqdm import tqdm
import numpy as np


def retrieve_knn(query_ids: List[str], key_ids: List[str], query_vecs, key_vecs, k=2047, query_batch_size=1000,
                 key_batch_size=10000):
    """
    Retrieve the top-k nearest neighbors for each query id from the key ids.
    Args:
        query_ids:
        key_ids:
        k: top-k
        query_batch_size:
        key_batch_size:

    Returns:

    """
    if len(key_vecs) == 0: return {}

    query_vecs = np.asarray(query_vecs, dtype=np.float32)
    key_vecs = np.asarray(key_vecs, dtype=np.float32)
    if query_vecs.ndim != 2 or key_vecs.ndim != 2:
        return {}

    def _normalize(x: np.ndarray) -> np.ndarray:
        denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return x / denom

    query_vecs = _normalize(query_vecs)
    key_vecs = _normalize(key_vecs)

    results = {}

    def get_batches(vecs, batch_size):
        for i in range(0, len(vecs), batch_size):
            yield vecs[i:i + batch_size], i

    for query_batch, query_batch_start_idx in tqdm(
            get_batches(vecs=query_vecs, batch_size=query_batch_size),
            total=(len(query_vecs) + query_batch_size - 1) // query_batch_size,  # Calculate total batches
            desc="KNN for Queries"
    ):
        # Maintain running top-k across key batches.
        topk_scores = None
        topk_indices = None
        offset_keys = 0

        for key_batch, _ in get_batches(vecs=key_vecs, batch_size=key_batch_size):
            sim = query_batch @ key_batch.T  # cosine sim due to normalization
            kk = min(k, sim.shape[1])
            part_idx = np.argpartition(sim, -kk, axis=1)[:, -kk:]
            part_scores = np.take_along_axis(sim, part_idx, axis=1)
            # sort within the partial topk
            order = np.argsort(part_scores, axis=1)[:, ::-1]
            part_idx = np.take_along_axis(part_idx, order, axis=1) + offset_keys
            part_scores = np.take_along_axis(part_scores, order, axis=1)

            if topk_scores is None:
                topk_scores = part_scores
                topk_indices = part_idx
            else:
                merged_scores = np.concatenate([topk_scores, part_scores], axis=1)
                merged_indices = np.concatenate([topk_indices, part_idx], axis=1)
                kk2 = min(k, merged_scores.shape[1])
                sel = np.argpartition(merged_scores, -kk2, axis=1)[:, -kk2:]
                topk_scores = np.take_along_axis(merged_scores, sel, axis=1)
                topk_indices = np.take_along_axis(merged_indices, sel, axis=1)
                order2 = np.argsort(topk_scores, axis=1)[:, ::-1]
                topk_scores = np.take_along_axis(topk_scores, order2, axis=1)
                topk_indices = np.take_along_axis(topk_indices, order2, axis=1)

            offset_keys += key_batch.shape[0]

        if topk_scores is None or topk_indices is None:
            continue

        for i in range(topk_indices.shape[0]):
            query_relative_idx = query_batch_start_idx + i
            query_idx = query_ids[query_relative_idx]
            key_idx_list = topk_indices[i].tolist()
            score_list = topk_scores[i].tolist()
            query_to_topk_key_ids = [key_ids[idx] for idx in key_idx_list]
            results[query_idx] = (query_to_topk_key_ids, score_list)
    # end for each query batch

    return results
