"""Step 1 of GVECF: pre-train a (V)GAE and save the variational embeddings.

Example:
    python pretrain_vgae.py --dataset gowalla --model vgae --embed_size 64
    python pretrain_vgae.py --dataset gowalla --model gae   # ablation (Table 3)

The output ``embeddings/<dataset>_<model>_d<embed_size>.npz`` holds
``user_mu``, ``item_mu`` (and ``user_logstd``, ``item_logstd`` for the VGAE)
and is passed to ``train.py --init <model> --pretrained <file>``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from gvecf import GraphAutoEncoder, InteractionDataset, evaluate
from gvecf.utils import get_device, set_seed, to_torch_sparse


def parse_args():
    p = argparse.ArgumentParser(description="Pre-train VGAE / GAE embeddings for GVECF")
    p.add_argument("--data_dir", default="data", help="folder containing one sub-folder per dataset")
    p.add_argument("--dataset", default="gowalla")
    p.add_argument("--model", choices=["vgae", "gae"], default="vgae")
    p.add_argument("--embed_size", type=int, default=64, help="must match --embed_size of train.py")
    p.add_argument("--hidden_size", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=3, help="GCN layers incl. the mu/logstd layer")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8192, help="positive edges per step")
    p.add_argument("--kl_weight", type=float, default=None, help="default: 1 / (#users + #items)")
    p.add_argument("--eval_every", type=int, default=10, help="report Recall/NDCG of the raw VEs; 0 = off")
    p.add_argument("--out_dir", default="embeddings")
    p.add_argument("--seed", type=int, default=2023)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = get_device(args.device)

    data = InteractionDataset(Path(args.data_dir) / args.dataset)
    print(data.summary())
    n_nodes = data.n_users + data.n_items
    kl_weight = args.kl_weight if args.kl_weight is not None else 1.0 / n_nodes

    adj = to_torch_sparse(data.normalized_adjacency(self_loops=True)).to(device)
    model = GraphAutoEncoder(
        n_nodes,
        embed_dim=args.embed_size,
        hidden_dim=args.hidden_size,
        n_layers=args.n_layers,
        dropout=args.dropout,
        variational=args.model == "vgae",
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    edges = data.train_edges
    steps = int(np.ceil(len(edges) / args.batch_size))
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(edges))
        totals = np.zeros(3)
        for s in range(steps):
            batch = edges[order[s * args.batch_size : (s + 1) * args.batch_size]]
            users, pos = batch[:, 0], batch[:, 1]
            neg = data.sample_negatives(users, rng)
            users, pos, neg = (torch.from_numpy(x).to(device) for x in (users, pos, neg))

            mu, logstd = model.encode(adj)
            z = model.reparameterize(mu, logstd)
            loss, recon, kl = model.loss(z, mu, logstd, users, pos, neg, data.n_users, kl_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totals += [loss.item(), recon.item(), kl.item()]

        totals /= steps
        msg = (
            f"epoch {epoch:4d} | {time.time() - start:7.1f}s | loss {totals[0]:.4f} "
            f"(recon {totals[1]:.4f}, kl {totals[2]:.4f})"
        )
        if args.eval_every and (epoch % args.eval_every == 0 or epoch == args.epochs):
            model.eval()
            with torch.no_grad():
                mu, _ = model.encode(adj)
            m = evaluate(mu[: data.n_users], mu[data.n_users :], data, [20])
            msg += f" | VE recall@20 {m['recall@20']:.4f} ndcg@20 {m['ndcg@20']:.4f}"
        print(msg, flush=True)

    pretrain_time = time.time() - start
    model.eval()
    with torch.no_grad():
        mu, logstd = model.encode(adj)
    mu = mu.cpu().numpy()
    arrays = {"user_mu": mu[: data.n_users], "item_mu": mu[data.n_users :]}
    if logstd is not None:
        logstd = logstd.cpu().numpy()
        arrays.update(user_logstd=logstd[: data.n_users], item_logstd=logstd[data.n_users :])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset}_{args.model}_d{args.embed_size}.npz"
    np.savez(
        out_path,
        **arrays,
        model=args.model,
        n_users=data.n_users,
        n_items=data.n_items,
        pretrain_time=pretrain_time,
    )
    print(f"saved {args.model.upper()} embeddings to {out_path} ({pretrain_time:.1f}s)")


if __name__ == "__main__":
    main()
