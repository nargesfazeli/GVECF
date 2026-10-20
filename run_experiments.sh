#!/usr/bin/env bash
# Full GVECF pipeline: pre-train VGAE/GAE embeddings, then train NGCF, GVECF
# and the GAE ablation on every dataset.
#
# Set LR / REG / NODE_DROPOUT per dataset to the values selected by the grid
# search (lr in {5e-4,1e-4,5e-5,1e-5,5e-6}, reg in {5e-4,1e-5,1e-6},
# node dropout in {0.1,0.2}).
set -euo pipefail

DATASETS=(gowalla yelp2018 amazon-book)
EMBED=64

declare -A LR=([gowalla]=1e-4 [yelp2018]=1e-4 [amazon-book]=1e-4)
declare -A REG=([gowalla]=1e-5 [yelp2018]=1e-5 [amazon-book]=1e-5)
declare -A NODE_DROPOUT=([gowalla]=0.1 [yelp2018]=0.1 [amazon-book]=0.1)

for ds in "${DATASETS[@]}"; do
  echo "=== ${ds} ==="
  python pretrain_vgae.py --dataset "$ds" --model vgae --embed_size $EMBED
  python pretrain_vgae.py --dataset "$ds" --model gae  --embed_size $EMBED

  common=(--dataset "$ds" --embed_size $EMBED --lr "${LR[$ds]}" --reg "${REG[$ds]}"
          --node_dropout "${NODE_DROPOUT[$ds]}")

  python train.py "${common[@]}" --run_name "${ds}_ngcf"
  python train.py "${common[@]}" --run_name "${ds}_gvecf_vgae" \
      --pretrained "embeddings/${ds}_vgae_d${EMBED}.npz"
  python train.py "${common[@]}" --run_name "${ds}_gvecf_gae" \
      --pretrained "embeddings/${ds}_gae_d${EMBED}.npz"

  python scripts/plot_curves.py "runs/${ds}_ngcf" "runs/${ds}_gvecf_vgae" "runs/${ds}_gvecf_gae" \
      --labels NGCF GVECF GVECF-GAE --out "figures/${ds}.png"
done
