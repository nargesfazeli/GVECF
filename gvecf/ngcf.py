"""Neural Graph Collaborative Filtering with optional variational initialisation.

Embedding propagation follows Eq. (10) of the paper (and NGCF, Wang et al.
SIGIR 2019):

    Z^(l) = LeakyReLU( (L + I) Z^(l-1) W1^(l) + (L Z^(l-1) ⊙ Z^(l-1)) W2^(l) )

The final representation concatenates every layer (Eq. 11) and the score is
the inner product of the user and item representations (Eq. 12).

The only difference between NGCF and GVECF is Z^(0): NGCF draws it with the
Xavier initialiser, while GVECF starts from the variational embeddings (VEs)
produced by the pre-trained VGAE (see ``pretrain_vgae.py``).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import sparse_dropout


class NGCF(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        laplacian: torch.Tensor,
        embed_dim: int = 64,
        layer_sizes: list[int] = (64, 64, 64),
        node_dropout: float = 0.1,
        mess_dropout: list[float] = (0.1, 0.1, 0.1),
        init_user_emb: np.ndarray | None = None,
        init_item_emb: np.ndarray | None = None,
    ):
        super().__init__()
        if len(mess_dropout) != len(layer_sizes):
            raise ValueError("mess_dropout needs one value per propagation layer")
        self.n_users, self.n_items = n_users, n_items
        self.laplacian = laplacian  # L = D^{-1/2} A D^{-1/2}, sparse
        self.node_dropout = node_dropout
        self.mess_dropout = list(mess_dropout)

        self.user_emb = nn.Parameter(torch.empty(n_users, embed_dim))
        self.item_emb = nn.Parameter(torch.empty(n_items, embed_dim))
        if init_user_emb is None or init_item_emb is None:
            nn.init.xavier_uniform_(self.user_emb)
            nn.init.xavier_uniform_(self.item_emb)
        else:
            self._load_initial(self.user_emb, init_user_emb, "user")
            self._load_initial(self.item_emb, init_item_emb, "item")

        dims = [embed_dim, *layer_sizes]
        self.W1 = nn.ModuleList(nn.Linear(dims[k], dims[k + 1]) for k in range(len(layer_sizes)))
        self.W2 = nn.ModuleList(nn.Linear(dims[k], dims[k + 1]) for k in range(len(layer_sizes)))
        for layer in [*self.W1, *self.W2]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    @staticmethod
    def _load_initial(param: nn.Parameter, values: np.ndarray, what: str) -> None:
        if tuple(values.shape) != tuple(param.shape):
            raise ValueError(
                f"pre-trained {what} embeddings have shape {values.shape}, "
                f"expected {tuple(param.shape)} (check dataset and --embed_size)"
            )
        with torch.no_grad():
            param.copy_(torch.as_tensor(values, dtype=param.dtype))

    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Run all propagation layers and return the final user/item representations."""
        L = self.laplacian
        if self.training and self.node_dropout > 0:
            L = sparse_dropout(L, self.node_dropout)

        ego = torch.cat([self.user_emb, self.item_emb], dim=0)
        layers = [ego]
        for k, (W1, W2) in enumerate(zip(self.W1, self.W2)):
            side = torch.sparse.mm(L, ego)          # L Z
            sum_emb = W1(side + ego)                # (L + I) Z W1
            bi_emb = W2(side * ego)                 # (L Z ⊙ Z) W2
            ego = F.leaky_relu(sum_emb + bi_emb, negative_slope=0.2)
            ego = F.dropout(ego, self.mess_dropout[k], self.training)
            layers.append(F.normalize(ego, p=2, dim=1))

        final = torch.cat(layers, dim=1)            # e* = e(0) || ... || e(L)
        return final[: self.n_users], final[self.n_users :]

    def bpr_loss(self, users, pos_items, neg_items, reg_weight: float):
        user_all, item_all = self.propagate()
        u, p, n = user_all[users], item_all[pos_items], item_all[neg_items]
        pos_scores = (u * p).sum(dim=1)
        neg_scores = (u * n).sum(dim=1)
        mf_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        reg = (u.pow(2).sum() + p.pow(2).sum() + n.pow(2).sum()) / 2
        emb_loss = reg_weight * reg / users.shape[0]
        return mf_loss + emb_loss, mf_loss, emb_loss
