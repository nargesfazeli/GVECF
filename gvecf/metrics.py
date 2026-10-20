"""Full-ranking top-K evaluation: Recall@K and NDCG@K.

For every test user all items are scored, items seen in training are
excluded, and the metrics are averaged over test users (same protocol as
NGCF / LightGCN).
"""

from __future__ import annotations

import numpy as np
import torch

from .data import InteractionDataset


@torch.no_grad()
def evaluate(
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    dataset: InteractionDataset,
    Ks: list[int] = (20,),
    batch_size: int = 2048,
) -> dict[str, float]:
    Ks = sorted(Ks)
    max_k = Ks[-1]
    test_users = np.asarray(sorted(dataset.test_items), dtype=np.int64)
    discounts = 1.0 / np.log2(np.arange(2, max_k + 2))

    recall = {k: 0.0 for k in Ks}
    ndcg = {k: 0.0 for k in Ks}

    for start in range(0, len(test_users), batch_size):
        users = test_users[start : start + batch_size]
        scores = user_emb[torch.from_numpy(users).to(user_emb.device)] @ item_emb.T

        # Mask training items so they are never recommended again.
        seen = dataset.R[users].tocoo()
        if seen.nnz:
            rows = torch.from_numpy(seen.row.astype(np.int64)).to(scores.device)
            cols = torch.from_numpy(seen.col.astype(np.int64)).to(scores.device)
            scores[rows, cols] = -np.inf

        top = torch.topk(scores, max_k, dim=1).indices.cpu().numpy()

        for row, u in enumerate(users):
            truth = dataset.test_items[u]
            hits = np.isin(top[row], truth).astype(np.float64)
            for k in Ks:
                h = hits[:k]
                recall[k] += h.sum() / len(truth)
                idcg = discounts[: min(len(truth), k)].sum()
                ndcg[k] += (h * discounts[:k]).sum() / idcg

    n = len(test_users)
    out = {}
    for k in Ks:
        out[f"recall@{k}"] = recall[k] / n
        out[f"ndcg@{k}"] = ndcg[k] / n
    return out
