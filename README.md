# Orthogonal Classifiers

Implementations of  [*Controlling Directions Orthogonal to a Classifier*](https://openreview.net/forum?id=DIjCrlsu6Z) , ICLR 2022

​												[Yilun Xu](yilun-xu.com), [Hao He](http://people.csail.mit.edu/hehaodele/), [Tianxiao Shen](https://people.csail.mit.edu/tianxiao/), [Tommi Jaakkola](http://people.csail.mit.edu/tommi/tommi.html)

Let's construct orthogonal classifiers for *controlled style transfer*, *domain adaptation with label shifts* and *fairness* problems :cowboy_hat_face: !



## Controlled Style Transfer

#### Prepare Celeba-GH dataset:

```shell
python style_transfer/celeba_dataset.py --data_dir {path}

path: path to the CelebA dataset
```

bash example: `python style_transfer/celeba_dataset.py --data_dir ./data`

You can modify the domain_fn dictionary in the `style_transfer/celeba_dataset.py` file to create new groups :bulb:



#### Step 1: Train $w_1,w_x$ and the ground truth $w_2$ (for measuring $z_2$ accuracy)

```shell
sh style_transfer/train_classifiers.sh {gpu} {path} {dataset} {alg}

gpu: the number of gpu
path: path to the dataset (Celeba or MNIST)
dataset: dataset (Celeba | CMNIST)
alg: ERM, Fish, TRM or MLDG
```

CMNIST bash example: `sh style_transfer/train_classifiers.sh 0 ./data CMNIST ERM`



#### Step 2: Train controlled CycleGAN

```shell
python style_transfer/train_cyclegan.py --data_dir {path} --dataset {dataset} --obj {obj} --name {name}

path: path to the dataset (Celeba or MNIST)
dataset: dataset (Celeba | CMNIST)
obj: training objective (vanilla | orthogonal)
name: name of the model
```

CMNIST bash example: `python style_transfer/train_cyclegan.py --data_dir ./data --dataset CMNIST --obj orthogonal --name cmnist`

To view training results and loss plots, run `python -m visdom.server` and click the URL [http://localhost:8097](http://localhost:8097/)



#### Evaluation and Generation

```shell
python style_transfer/generate.py --data_dir {path} --dataset {dataset} --name {name} --obj {obj} --out_path {out_path} --resume_epoch {epoch} (--save)

path: path to the dataset (Celeba or MNIST)
dataset: dataset (Celeba | CMNIST)
name: name of the model
obj: training objective (vanilla | orthogonal)
out_path: output path
epoch: resuming epoch of checkpoint
```

Images will be save to `style_transfer/generated_images/out_path`

CMNIST bash example: `python style_transfer/generate.py --data_dir ./data --dataset CMNIST --name cmnist --obj orthogonal --out_path cmnist_out --resume_epoch 5`



## Domain Adaptation (DA) with label shifts

#### Prepare src/tgt pairs with label shifts

```shell

```



#### Training

```shell
python da/vada_train.py --r 0.7 0.3 --src {source} --tgt {target}  --seed {seed} (--iw) (--orthogonal) (--source_only)

source: source domain (mnist | mnistm | svhn | cifar | stl | signs)
target: target domain (mnist | mnistm | svhn | cifar | stl | signs)
seed: random seed
--source_only: vanilla ERM on the source domain
--iw: use importance-weighted domain adaptation
--orth: use orthogonal classifier
```





## Fairness

```

```











