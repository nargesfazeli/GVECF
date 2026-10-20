"""Step 2 of GVECF: train NGCF from Xavier (baseline) or variational embeddings.

Examples:
    # Baseline NGCF (Xavier initialisation)
    python train.py --dataset gowalla

    # GVECF (NGCF initialised with VGAE embeddings)
    python train.py --dataset gowalla --pretrained embeddings/gowalla_vgae_d64.npz

    # Ablation: NGCF initialised with GAE embeddings
    python train.py --dataset gowalla --pretrained embeddings/gowalla_gae_d64.npz

Every run writes ``runs/<name>/`` with the config, a per-evaluation CSV log
(used by ``scripts/plot_curves.py``), the best checkpoint and results.json.
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from gvecf import NGCF, InteractionDataset, evaluate
from gvecf.utils import get_device, get_logger, save_json, set_seed, to_torch_sparse


def parse_args():
    p = argparse.ArgumentParser(description="Train NGCF / GVECF")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--dataset", default="gowalla")
    p.add_argument("--pretrained", default=None, help=".npz from pretrain_vgae.py; omit for Xavier (NGCF)")
    p.add_argument("--ve_mode", choices=["mu", "sample"], default="mu",
                   help="initialise with the posterior mean, or with one sample z = mu + sigma * eps")
    p.add_argument("--embed_size", type=int, default=64)
    p.add_argument("--layer_size", type=int, nargs="+", default=[64, 64, 64])
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--reg", type=float, default=1e-5, help="L2 regularisation strength (lambda_reg)")
    p.add_argument("--node_dropout", type=float, default=0.1)
    p.add_argument("--mess_dropout", type=float, nargs="+", default=[0.1, 0.1, 0.1])
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--patience", type=int, default=5, help="evaluations without improvement before stopping")
    p.add_argument("--Ks", type=int, nargs="+", default=[20])
    p.add_argument("--test_batch_size", type=int, default=2048)
    p.add_argument("--run_name", default=None)
    p.add_argument("--out_dir", default="runs")
    p.add_argument("--seed", type=int, default=2023)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def load_pretrained(path: str, mode: str, data: InteractionDataset, seed: int):
    ve = np.load(path)
    if int(ve["n_users"]) != data.n_users or int(ve["n_items"]) != data.n_items:
        raise ValueError(f"{path} was pre-trained on a different dataset")
    user, item = ve["user_mu"], ve["item_mu"]
    if mode == "sample":
        if "user_logstd" not in ve:
            raise ValueError("--ve_mode sample needs VGAE embeddings (GAE has no variance)")
        rng = np.random.default_rng(seed)
        user = user + np.exp(ve["user_logstd"]) * rng.standard_normal(user.shape)
        item = item + np.exp(ve["item_logstd"]) * rng.standard_normal(item.shape)
    return user.astype(np.float32), item.astype(np.float32), str(ve["model"]), float(ve["pretrain_time"])


def main():
    args = parse_args()
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = get_device(args.device)
    Ks = sorted(args.Ks)
    key_metric = f"recall@{Ks[0]}"

    data = InteractionDataset(Path(args.data_dir) / args.dataset)

    init_user = init_item = None
    init_name, pretrain_time = "xavier", 0.0
    if args.pretrained:
        init_user, init_item, init_name, pretrain_time = load_pretrained(
            args.pretrained, args.ve_mode, data, args.seed
        )
    method = "ngcf" if init_name == "xavier" else f"gvecf_{init_name}"

    run_name = args.run_name or f"{args.dataset}_{method}_{datetime.now():%Y%m%d-%H%M%S}"
    run_dir = Path(args.out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger(run_dir)
    save_json(vars(args) | {"method": method}, run_dir / "config.json")
    log.info(data.summary())
    log.info(f"method={method} device={device} init={init_name}"
             + (f" ({args.ve_mode}, pre-training took {pretrain_time:.1f}s)" if args.pretrained else ""))

    laplacian = to_torch_sparse(data.normalized_adjacency(self_loops=False)).to(device)
    model = NGCF(
        data.n_users,
        data.n_items,
        laplacian,
        embed_dim=args.embed_size,
        layer_sizes=args.layer_size,
        node_dropout=args.node_dropout,
        mess_dropout=args.mess_dropout,
        init_user_emb=init_user,
        init_item_emb=init_item,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    csv_path = run_dir / "log.csv"
    fields = ["epoch", "train_time", "time_incl_pretrain", "loss", "mf_loss", "emb_loss"]
    fields += [f"{m}@{k}" for m in ("recall", "ndcg") for k in Ks]
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(fields)

    n_batch = data.n_train // args.batch_size + 1
    train_time, best, best_epoch, bad_evals = 0.0, None, 0, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        totals = np.zeros(3)
        for _ in range(n_batch):
            users, pos, neg = (torch.from_numpy(x).to(device)
                               for x in data.sample_bpr_batch(args.batch_size, rng))
            loss, mf_loss, emb_loss = model.bpr_loss(users, pos, neg, args.reg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totals += [loss.item(), mf_loss.item(), emb_loss.item()]
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_time += time.time() - t0
        totals /= n_batch

        if epoch % args.eval_every and epoch != args.epochs:
            continue

        model.eval()
        with torch.no_grad():
            user_all, item_all = model.propagate()
        metrics = evaluate(user_all, item_all, data, Ks, args.test_batch_size)

        row = [epoch, round(train_time, 2), round(train_time + pretrain_time, 2), *np.round(totals, 6)]
        row += [round(metrics[f], 6) for f in fields[6:]]
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        log.info(
            f"epoch {epoch:4d} | {train_time:8.1f}s | loss {totals[0]:.5f} | "
            + " ".join(f"{k} {v:.5f}" for k, v in metrics.items())
        )

        if best is None or metrics[key_metric] > best[key_metric]:
            best, best_epoch, bad_evals = metrics, epoch, 0
            best_time = train_time
            torch.save(model.state_dict(), run_dir / "best.pt")
        else:
            bad_evals += 1
            if bad_evals >= args.patience:
                log.info(f"early stopping: no {key_metric} improvement in {args.patience} evaluations")
                break

    results = {
        "method": method,
        "dataset": args.dataset,
        "best_epoch": best_epoch,
        "train_time_to_best_sec": round(best_time, 2),
        "pretrain_time_sec": round(pretrain_time, 2),
        **{k: round(v, 6) for k, v in best.items()},
    }
    save_json(results, run_dir / "results.json")
    log.info("best: " + " ".join(f"{k}={v}" for k, v in results.items()))


if __name__ == "__main__":
    main()
