"""Variational graph auto-encoder (VGAE) used to pre-train the embeddings.

Encoder (inference model, Eqs. 2-3): a GCN over the user-item graph that
outputs a mean ``mu`` and a log standard deviation ``log sigma`` for every
user and item node. Users and items have no side features, so the input
feature matrix is the identity X = I and the first GCN weight matrix is
therefore a learnable node-embedding table.

Decoder (generative model, Eq. 4): p(A_ui = 1 | z_u, z_i) = sigmoid(z_u^T z_i).
Because the full (N+M) x (N+M) reconstruction does not fit in memory for
real datasets, the reconstruction loss is computed on observed user-item
edges and an equal number of sampled non-edges.

Setting ``variational=False`` gives a plain graph auto-encoder (GAE) with
the same structure, used as the ablation in Table 3 of the paper.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_LOGSTD = 10.0


class GraphAutoEncoder(nn.Module):
    def __init__(
        self,
        n_nodes: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        n_layers: int = 3,
        dropout: float = 0.0,
        variational: bool = True,
    ):
        """
        Args:
            n_nodes: number of users + items.
            embed_dim: size of the latent variational embeddings (VEs).
            hidden_dim: width of the hidden GCN layers.
            n_layers: total GCN layers, including the input layer and the
                mu / log-sigma output layer (the paper uses 3).
            dropout: dropout applied after each hidden layer.
            variational: VGAE if True, GAE if False.
        """
        super().__init__()
        if n_layers < 2:
            raise ValueError("n_layers must be at least 2")
        self.variational = variational
        self.dropout = dropout

        # Layer 1: A_hat X W0 with X = I  ->  A_hat W0, i.e. an embedding table.
        self.input_weight = nn.Parameter(torch.empty(n_nodes, hidden_dim))
        self.hidden = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 2)
        )
        self.mu_head = nn.Linear(hidden_dim, embed_dim)
        self.logstd_head = nn.Linear(hidden_dim, embed_dim) if variational else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_weight)
        for layer in [*self.hidden, self.mu_head] + ([self.logstd_head] if self.variational else []):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    # ---------------------------------------------------------------- encoder
    def encode(self, adj: torch.Tensor):
        h = torch.sparse.mm(adj, self.input_weight)
        h = F.dropout(F.relu(h), self.dropout, self.training)
        for layer in self.hidden:
            h = torch.sparse.mm(adj, layer(h))
            h = F.dropout(F.relu(h), self.dropout, self.training)
        mu = torch.sparse.mm(adj, self.mu_head(h))
        if not self.variational:
            return mu, None
        logstd = torch.sparse.mm(adj, self.logstd_head(h)).clamp(max=MAX_LOGSTD)
        return mu, logstd

    def reparameterize(self, mu: torch.Tensor, logstd: torch.Tensor | None) -> torch.Tensor:
        if logstd is None or not self.training:
            return mu
        return mu + torch.randn_like(mu) * torch.exp(logstd)

    # ---------------------------------------------------------------- decoder
    @staticmethod
    def decode(z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        """Logits of p(A_ij = 1 | z_i, z_j) = sigmoid(z_i^T z_j)."""
        return (z_src * z_dst).sum(dim=-1)

    # ------------------------------------------------------------------- loss
    def loss(
        self,
        z: torch.Tensor,
        mu: torch.Tensor,
        logstd: torch.Tensor | None,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        n_users: int,
        kl_weight: float,
    ):
        """Negative ELBO: reconstruction BCE + weighted KL(q(z|X,A) || N(0, I)).

        Item indices are item ids (0..M-1); they are offset by ``n_users`` to
        index the joint user/item node table.
        """
        z_u = z[users]
        pos_logits = self.decode(z_u, z[pos_items + n_users])
        neg_logits = self.decode(z_u, z[neg_items + n_users])
        recon = F.binary_cross_entropy_with_logits(
            pos_logits, torch.ones_like(pos_logits)
        ) + F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))

        if logstd is None:
            return recon, recon, torch.zeros((), device=z.device)
        kl = -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu.pow(2) - torch.exp(logstd).pow(2), dim=1)
        )
        return recon + kl_weight * kl, recon, kl
