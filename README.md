# Orthogonal Classifiers

Implementations of paper [*Controlling Directions Orthogonal to a Classifier*](https://openreview.net/forum?id=DIjCrlsu6Z) , ICLR 2022,  &nbsp;[Yilun Xu](yilun-xu.com), [Hao He](http://people.csail.mit.edu/hehaodele/), [Tianxiao Shen](https://people.csail.mit.edu/tianxiao/), [Tommi Jaakkola](http://people.csail.mit.edu/tommi/tommi.html)

Let's construct orthogonal classifiers for *controlled style transfer*, *domain adaptation with label shifts* and *fairness* problems :cowboy_hat_face: !

## Outline

- [Controlled Style Transfer](#style)
  - [Prepare Celeba-GH dataset](#celebagh)
  - Train [classifiers](#classifier) and [CycleGAN](#cyclegan)
- [Domain Adaptation with label shifts](#da)
  - [Prepare dataset pairs](#da.data)
  - [Training](#da.train)

- [Fairness](#fair)

## Controlled Style Transfer<a id="style"></a>

#### Prepare Celeba-GH dataset<a id="celebagh"></a>:

```shell
python style_transfer/celeba_dataset.py --data_dir {path}

path: path to the CelebA dataset
```

bash example: `python style_transfer/celeba_dataset.py --data_dir ./data`

One can modify the `domain_fn` dictionary in the `style_transfer/celeba_dataset.py` file to create new groups :bulb:



#### Step 1: Train principal, full and oracle orthogonal classifiers <a id="classifier"></a>

```shell
sh style_transfer/train_classifiers.sh {gpu} {path} {dataset} {alg}

gpu: the number of gpu
path: path to the dataset (Celeba or MNIST)
dataset: dataset (Celeba | CMNIST)
alg: ERM, Fish, TRM or MLDG
```

CMNIST bash example: `sh style_transfer/train_classifiers.sh 0 ./data CMNIST ERM`



#### Step 2: Train controlled CycleGAN<a id="cyclegan"></a>

```shell
python style_transfer/train_cyclegan.py --data_dir {path} --dataset {dataset} \
  --obj {obj} --name {name}

path: path to the dataset (Celeba or MNIST)
dataset: dataset (Celeba | CMNIST)
obj: training objective (vanilla | orthogonal)
name: name of the model
```

CMNIST bash example: `python style_transfer/train_cyclegan.py --data_dir ./data --dataset CMNIST --obj orthogonal --name cmnist`

To view training results and loss plots, run `python -m visdom.server` and click the URL [http://localhost:8097](http://localhost:8097/)



#### Evaluation and Generation

```shell
python style_transfer/generate.py --data_dir {path} --dataset {dataset} --name {name} \
 --obj {obj} --out_path {out_path} --resume_epoch {epoch} (--save)

path: path to the dataset (Celeba or MNIST)
dataset: dataset (Celeba | CMNIST)
name: name of the model
obj: training objective (vanilla | orthogonal)
out_path: output path
epoch: resuming epoch of checkpoint
```

Images will be save to `style_transfer/generated_images/out_path`

CMNIST bash example: `python style_transfer/generate.py --data_dir ./data --dataset CMNIST --name cmnist --obj orthogonal --out_path cmnist_out --resume_epoch 5`

<br/>

## Domain Adaptation (DA) with label shifts<a id="da"></a>

#### Prepare src/tgt pairs with label shifts<a id="da.data"></a>

Please `cd /da/data` and run

```shell
python {dataset}.py --r {r0} {r1}

r0: subsample ratio for the first half classes (default=0.7)
r1: subsample ratio for the first half classes (default=0.3)
dataset: mnist | mnistm | svhn | cifar | stl | signs | digits
```

For *SynthDigits* / *SynthSigns*dataset, please download them at [link_digits](https://drive.google.com/uc?id=0B9Z4d7lAwbnTSVR1dEFSRUFxOUU)  / [link_signs](https://drive.google.com/open?id=1wgLzFwrUOz0dLjuCWZ0ylDWQR0xTdJ9X). All the other datasets will be automatically downloaded 😉



#### Training<a id="da.train"></a>

```shell
python da/vada_train.py --r {r0} {r1} --src {source} --tgt {target}  --seed {seed} \
 (--iw) (--orthogonal) (--source_only)

r0: subsample ratio for the first half classes (default=0.7)
r1: subsample ratio for the first half classes (default=0.3)
source: source domain (mnist | mnistm | svhn | cifar | stl | signs | digits)
target: target domain (mnist | mnistm | svhn | cifar | stl | signs | digits)
seed: random seed
--source_only: vanilla ERM on the source domain
--iw: use importance-weighted domain adaptation algorithm [1]
--orthogonal: use orthogonal classifier
--vada: vanilla VADA [2]
```

<br/>

## Fairness<a id="fair"></a>

```shell
python fairness/methods/train.py --data {data} --gamma {gamma} --sigma {sigma} \
 (--orthogonal) (--laftr) (--mifr) (--hsic)

data: dataset (adult | german)
gamma: hyper-parameter for MIFR, HSIC, LAFTR
sigma: hyper-parameter for HSIC (kernel width)
--orthogonal: use orthogonal classifier
--MIFR: use L-MIFR algorithm [3]
--HSIC: use ReBias algorithm [4]
--LAFTR: use LAFTR algorithm [5]
```



---



<br/>

<br/>

#### Reference

**\[1\]** Remi Tachet des Combes, Han Zhao, Yu-Xiang Wang, and Geoffrey J. Gordon. *Domain adaptation with conditional distribution matching and generalized label shift*. *ArXiv*, abs/2003.04475, 2020.

**\[2\]** Rui Shu, H. Bui, H. Narui, and S. Ermon. A dirt-t approach to unsupervised domain adaptation. *ArXiv*, abs/1802.08735, 2018.

**\[3\]** Jiaming Song, Pratyusha Kalluri, Aditya Grover, Shengjia Zhao, and S. Ermon. Learning controllable fair representations. In *AISTATS*, 2019.

**\[4\]** Hyojin Bahng, Sanghyuk Chun, Sangdoo Yun, Jaegul Choo, and Seong Joon Oh. Learning de-biased representations with biased representations. In *ICML*, 2020.

**\[5\]** David Madras, Elliot Creager, T. Pitassi, and R. Zemel. Learning adversarially fair and transferable representations. In *ICML*, 2018.

<br/>

The implementation of this repo is based on / inspired by:

- https://github.com/facebookresearch/DomainBed (code structure).
- https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix (code structure)
- https://github.com/ozanciga/dirt-t (VADA code)
- https://github.com/Britefury/self-ensemble-visual-domain-adapt (data generation)