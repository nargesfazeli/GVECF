"""Create a small synthetic dataset in NGCF format for a quick smoke test.

Users and items belong to latent communities; users mostly interact with
items of their own community. The result is written to data/toy/.

    python scripts/make_toy_data.py
    python pretrain_vgae.py --dataset toy --epochs 20 --batch_size 2048
    python train.py --dataset toy --epochs 40 --pretrained embeddings/toy_vgae_d64.npz
"""

import argparse
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/toy")
    p.add_argument("--users", type=int, default=600)
    p.add_argument("--items", type=int, default=800)
    p.add_argument("--communities", type=int, default=8)
    p.add_argument("--per_user", type=int, default=25)
    p.add_argument("--test_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    user_c = rng.integers(0, args.communities, args.users)
    item_c = rng.integers(0, args.communities, args.items)
    popularity = rng.pareto(1.5, args.items) + 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "train.txt", "w") as ftr, open(out / "test.txt", "w") as fte:
        for u in range(args.users):
            weight = popularity * np.where(item_c == user_c[u], 20.0, 1.0)
            n = max(5, int(rng.poisson(args.per_user)))
            items = rng.choice(args.items, size=n, replace=False, p=weight / weight.sum())
            rng.shuffle(items)
            n_test = max(1, int(round(n * args.test_ratio)))
            ftr.write(" ".join(map(str, [u, *sorted(items[n_test:])])) + "\n")
            fte.write(" ".join(map(str, [u, *sorted(items[:n_test])])) + "\n")
    print(f"wrote {args.users} users / {args.items} items to {out}")


if __name__ == "__main__":
    main()
