"""Loading implicit-feedback data and building the user-item graph.

The expected format is the one used by NGCF / LightGCN: a folder with
``train.txt`` and ``test.txt``, where every line is

    user_id item_id_1 item_id_2 ...

with 0-based integer ids.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp


def _read_interactions(path: Path) -> dict[int, np.ndarray]:
    interactions: dict[int, np.ndarray] = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            user = int(parts[0])
            items = np.asarray([int(x) for x in parts[1:]], dtype=np.int64)
            if items.size:
                interactions[user] = np.unique(items)
    return interactions


class InteractionDataset:
    """User-item interactions with helpers for graphs and BPR sampling."""

    def __init__(self, data_dir: str | Path):
        data_dir = Path(data_dir)
        self.name = data_dir.name
        self.train_items = _read_interactions(data_dir / "train.txt")
        self.test_items = _read_interactions(data_dir / "test.txt")

        all_users = list(self.train_items) + list(self.test_items)
        all_items = [i for d in (self.train_items, self.test_items) for v in d.values() for i in v]
        self.n_users = max(all_users) + 1
        self.n_items = int(max(all_items)) + 1

        rows = np.concatenate([np.full(len(v), u) for u, v in self.train_items.items()])
        cols = np.concatenate(list(self.train_items.values()))
        self.train_edges = np.stack([rows, cols], axis=1).astype(np.int64)
        self.n_train = len(self.train_edges)
        self.n_test = sum(len(v) for v in self.test_items.values())

        # R: user-item interaction matrix (N x M), Eq. (1) of the paper.
        self.R = sp.csr_matrix(
            (np.ones(self.n_train, dtype=np.float32), (rows, cols)),
            shape=(self.n_users, self.n_items),
        )
        self._train_users = np.asarray(sorted(self.train_items), dtype=np.int64)
        self._degree = np.diff(self.R.indptr)
        # Sorted keys u * M + i, used for fast "is this a training edge?" checks.
        self._edge_keys = np.sort(rows * self.n_items + cols)

    # ------------------------------------------------------------------ graph
    def adjacency(self) -> sp.csr_matrix:
        """Bipartite adjacency A = [[0, R], [R^T, 0]] of size (N+M) x (N+M)."""
        R = self.R.tocoo()
        n = self.n_users + self.n_items
        row = np.concatenate([R.row, R.col + self.n_users])
        col = np.concatenate([R.col + self.n_users, R.row])
        data = np.ones(len(row), dtype=np.float32)
        return sp.csr_matrix((data, (row, col)), shape=(n, n))

    def normalized_adjacency(self, self_loops: bool = False) -> sp.csr_matrix:
        """Symmetric normalisation D^{-1/2} (A [+ I]) D^{-1/2}.

        ``self_loops=False`` gives the graph Laplacian L used by NGCF
        (L_ui = 1 / sqrt(|N_u||N_i|)); ``self_loops=True`` gives the
        renormalised adjacency used by the GCN encoder of the (V)GAE.
        """
        adj = self.adjacency()
        if self_loops:
            adj = adj + sp.eye(adj.shape[0], dtype=np.float32, format="csr")
        deg = np.asarray(adj.sum(axis=1)).ravel()
        with np.errstate(divide="ignore"):
            d_inv_sqrt = np.power(deg, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        D = sp.diags(d_inv_sqrt)
        return (D @ adj @ D).tocsr().astype(np.float32)

    # --------------------------------------------------------------- sampling
    def is_train_edge(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        keys = users * self.n_items + items
        pos = np.searchsorted(self._edge_keys, keys)
        pos = np.minimum(pos, len(self._edge_keys) - 1)
        return self._edge_keys[pos] == keys

    def sample_negatives(self, users: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """One item per user that the user has not interacted with in training."""
        neg = rng.integers(0, self.n_items, size=len(users))
        bad = self.is_train_edge(users, neg)
        while bad.any():
            neg[bad] = rng.integers(0, self.n_items, size=int(bad.sum()))
            bad = self.is_train_edge(users, neg)
        return neg

    def sample_bpr_batch(self, batch_size: int, rng: np.random.Generator):
        """(user, positive item, negative item) triples, as in NGCF."""
        users = rng.choice(self._train_users, size=batch_size)
        offsets = self.R.indptr[users] + (rng.random(batch_size) * self._degree[users]).astype(np.int64)
        pos = self.R.indices[offsets].astype(np.int64)
        neg = self.sample_negatives(users, rng)
        return users, pos, neg

    def summary(self) -> str:
        density = (self.n_train + self.n_test) / (self.n_users * self.n_items)
        return (
            f"{self.name}: {self.n_users} users, {self.n_items} items, "
            f"{self.n_train} train / {self.n_test} test interactions, "
            f"density {density:.5f}"
        )
