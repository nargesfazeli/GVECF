"""Plot training curves of several runs (similar to Figs. 2 and 4 of the paper).

    python scripts/plot_curves.py runs/gowalla_ngcf_* runs/gowalla_gvecf_vgae_* \
        --labels NGCF GVECF --out figures/gowalla.png

Left panel: NDCG@K vs. epoch. Right panel: elapsed training time vs. Recall@K.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read_log(run_dir: Path):
    with open(run_dir / "log.csv") as f:
        rows = list(csv.DictReader(f))
    return {k: [float(r[k]) for r in rows] for k in rows[0]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="run folders created by train.py")
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--include_pretrain", action="store_true",
                   help="add the VGAE pre-training time to the elapsed time")
    p.add_argument("--out", default="figures/curves.png")
    args = p.parse_args()

    labels = args.labels or [Path(r).name for r in args.runs]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    time_col = "time_incl_pretrain" if args.include_pretrain else "train_time"
    for run, label in zip(args.runs, labels):
        log = read_log(Path(run))
        ax1.plot(log["epoch"], log[f"ndcg@{args.k}"], marker="o", ms=3, label=label)
        ax2.plot(log[f"recall@{args.k}"], log[time_col], marker="o", ms=3, label=label)
    ax1.set(xlabel="Epoch", ylabel=f"NDCG@{args.k}")
    ax2.set(xlabel=f"Recall@{args.k}", ylabel="Elapsed time (s)")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
