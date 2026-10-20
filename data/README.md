# Datasets

Each dataset lives in its own folder with two files:

```
data/
├── gowalla/      train.txt  test.txt
├── yelp2018/     train.txt  test.txt
└── amazon-book/  train.txt  test.txt
```

Every line is a user followed by the items they interacted with (0-based ids, space separated):

```
0 0 1 2 3 4 5
1 10 11 12
```

These are the 10-core benchmark splits released with NGCF and LightGCN:

| Dataset | Source |
|---|---|
| Gowalla | [NGCF repository, `Data/gowalla`](https://github.com/xiangwang1223/neural_graph_collaborative_filtering/tree/master/Data/gowalla) |
| Amazon-Book | [NGCF repository, `Data/amazon-book`](https://github.com/xiangwang1223/neural_graph_collaborative_filtering/tree/master/Data/amazon-book) |
| Yelp2018 | [LightGCN-PyTorch repository, `data/yelp2018`](https://github.com/gusye1234/LightGCN-PyTorch/tree/master/data/yelp2018) |

Copy `train.txt` and `test.txt` of each dataset into the matching folder above. The data files are not tracked by git (see `.gitignore`).
