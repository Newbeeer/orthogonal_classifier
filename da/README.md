[![DOI](https://zenodo.org/badge/190050339.svg)](https://zenodo.org/badge/latestdoi/190050339)

# dirt-t

Pytorch implementation of the paper [A DIRT-T Approach to Unsupervised Domain Adaptation](https://arxiv.org/abs/1802.08735). The code here only partially mirrors the original work. It should be possible to use VADA model and a bit of code reuse from the script `vada_train.py` to be able to perform the recursive iteration described in the paper.

### *dependencies*:
```python
python==3.7
torch==1.0
tqdm==4.31
```

### *data*:

Go to the [official repo](https://github.com/RuiShu/dirt-t), data/ directory and use `download_mnist.py` and `download_svhn.py` to get required .mat files. Place them under `data/mnist/` and `data/svhn/` folders. Then running `python vada_train.py` should work.
