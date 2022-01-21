# Orthogonal Classifiers

[*Controlling Directions Orthogonal to a Classifier*](https://openreview.net/forum?id=DIjCrlsu6Z) , ICLR 2022

​		Yilun Xu, Hao He, Tianxiao Shen, Tommi Jaakkola

Let's construct orthogonal classifiers for *controlled style transfer*, *domain adaptation with label shifts* and *fairness* problems :cowboy_hat_face: !



## Controlled Style Transfer





## Domain Adaptation with label shifts





## Fairness



### Evaluate the orthogonal classifier 

Datasets: ColoredMNIST

- Orthogonal classifier:

  ```bash
  sh scripts/run.sh <bias>
  
  bias: The bias degree on the back-ground colors
  ```

- Oracle:

  ```bash
  python -m train_oracle   --data_dir=../domainbed \
  --algorithm ERM  --dataset ColoredMNIST  --trial_seed 0 --bias 0. bias --epochs 1
  
  bias: The bias degree on the back-ground colors
  ```

- Reweighed classifier:

  ```bash
  sh scripts/run_reweight.sh <bias>
  
  bias: The bias degree on the back-ground colors
  ```

  

### Style Transfer

- To view training results and loss plots, run `python -m visdom.server` and click the URL [http://localhost:8097](http://localhost:8097/)

- ColoredMNIST:

  - Train classifier: 

    
  
  ```shell
python train_cyclegan.py --model cycle_gan --pool_size 50 --no_dropout \
  --gan_mode vanilla --name vanilla --eval --netG mnist \
--netD mnist --dataset ColoredMNIST --bias 1. 0.6 0.8 --batch_size 128 --obj obj
  
  obj: vanilla | kl | js
  
  
  # example
  CUDA_VISIBLE_DEVICES=2 python train_cyclegan.py --dataroot ./datasets --model cycle_gan --pool_size 50 --no_dropout \
  --gan_mode vanilla --name mnist_js_mldg --eval --netG mnist \
  --netD mnist --dataset ColoredMNIST --bias 0.9 0.8 0.8 --batch_size 128 --obj js --alg MLDG
  
  CUDA_VISIBLE_DEVICES=2 python train_cyclegan.py --dataroot ./datasets --model cycle_gan --pool_size 50 --no_dropout \
  --gan_mode vanilla --name mnist_vanilla --eval --netG mnist \
  --netD mnist --dataset ColoredMNIST --bias 0.9 0.8 0.8 --batch_size 128 --obj vanilla --alg ERM
  
  CUDA_VISIBLE_DEVICES=2 python train_cyclegan.py --dataroot ./datasets --model cycle_gan --pool_size 50 --no_dropout \
  --gan_mode vanilla --name mnist_vanilla --eval --netG mnist \
  --netD mnist --dataset ColoredMNIST --bias 0.6 0.6 0.8 --batch_size 128 --obj vanilla
  
  CUDA_VISIBLE_DEVICES=2 python train_cyclegan.py --dataroot ./datasets --model cycle_gan --pool_size 50 --no_dropout \
  --gan_mode vanilla --name mnist_js --eval --netG mnist \
  --netD mnist --dataset ColoredMNIST --bias 0.6 0.6 0.8 --batch_size 128 --obj js
  
  
  CUDA_VISIBLE_DEVICES=1 python train_cyclegan.py --dataroot ./datasets/cmnist_0.6 --model cycle_gan --pool_size 50 --no_dropout \
  --gan_mode vanilla --eval --netG mnist \
  --netD mnist --dataset ColoredMNIST --name mnist_re --bias 0.9 0.8 0.8 --batch_size 128 --obj js --n_epochs 6 --reweight
  
  obj: vanilla | kl | js
  ```
  
  
  
  - Generate:

    ```shell
  CUDA_VISIBLE_DEVICES=3 python3 generate.py --dataroot ./datasets/cmnist_0.6 --netG mnist --gan_mode vanilla --model cycle_gan --netD mnist --dataset ColoredMNIST --name mnist_js --out_path mnist_js_6 --bias 0.9 0.8 0.8 --obj js --alg ERM --resume_epoch 1 --save
    
  
    CUDA_VISIBLE_DEVICES=3 python3 generate.py --dataroot ./datasets/cmnist_0.6 --netG mnist --gan_mode vanilla --model cycle_gan --netD mnist --dataset ColoredMNIST --name mnist_vanilla --out_path mnist_vanilla_3 --bias 1. 0.6 0.8 --obj vanilla --alg ERM --resume_epoch 6 --save
    
    
    CUDA_VISIBLE_DEVICES=3 python3 generate.py --dataroot ./datasets/cmnist_0.6 --netG mnist --gan_mode vanilla --model cycle_gan --netD mnist --dataset ColoredMNIST --name mnist_js_trm --out_path mnist_trm --bias 0.9 0.8 0.8 --obj js --alg TRM --resume_epoch 3 --save
    
    CUDA_VISIBLE_DEVICES=3 python3 generate.py --dataroot ./datasets/cmnist_0.6 --netG mnist --gan_mode vanilla --model cycle_gan --netD mnist --dataset ColoredMNIST --name mnist_js_mldg --out_path mnist_mldg --bias 0.9 0.8 0.8 --obj js --alg MLDG --resume_epoch 3 --save
    
    
    CUDA_VISIBLE_DEVICES=2 python3 generate.py --dataroot ./datasets/cmnist_0.6 --netG mnist --gan_mode vanilla --model cycle_gan --netD mnist --dataset ColoredMNIST --name mnist_js_fish --out_path mnist_fish --bias 0.9 0.8 0.8 --obj js --alg Fish --resume_epoch 3 --save
    ```
  
    
  
  

- CelebA:

  - Train classifiers

  ```shell
  # oracle
    
  CUDA_VISIBLE_DEVICES=3 python -m train_w1_dg  --data_dir=~/domainbed --algorithm TRM \
  --dataset Celeba  --trial_seed 1 --model_save --epochs 10 --stage 1
  
  
  
  CUDA_VISIBLE_DEVICES=2 python -m train_w1_dg  --data_dir=~/domainbed --algorithm ERM \
    --dataset Celeba  --trial_seed 1 --model_save --epochs 10 --stage 1
    
    CUDA_VISIBLE_DEVICES=3 python -m train_erm_copy  --data_dir=~/domainbed --algorithm ERM \
    --dataset Celeba  --trial_seed 1 --epochs 1 --stage 2 --resume
    
      CUDA_VISIBLE_DEVICES=3 python -m train_erm_copy  --data_dir=~/domainbed --algorithm ERM \
    --dataset Celeba  --trial_seed 1 --epochs 1 --stage 3 --resume
    
   CUDA_VISIBLE_DEVICES=3 python -m train_w1_dg  --data_dir=~/domainbed --algorithm TRM \
    --dataset Celeba  --trial_seed 1 --model_save --epochs 10 --age
   
    CUDA_VISIBLE_DEVICES=0 python -m train_w1_dg  --data_dir=~/domainbed --algorithm MLDG \
    --dataset Celeba  --trial_seed 1 --model_save --epochs 10 --age
    
    CUDA_VISIBLE_DEVICES=0 python -m train_w1_dg  --data_dir=~/domainbed --algorithm ERM \
    --dataset Celeba  --trial_seed 1 --model_save --epochs 10 --age
    
    
  Stage 1: invariant classifier
  Stage 2: Full classifier
  Stage 3: orthogonal classifier (oracle)
  
  # w_1
  CUDA_VISIBLE_DEVICES=1 python -m train_erm  --data_dir=~/domainbed --algorithm ERM \
    --dataset Celeba  --trial_seed 1 --model_save --epochs 10 --stage 1
  # w_x
  CUDA_VISIBLE_DEVICES=1 python -m train_erm  --data_dir=~/domainbed --algorithm ERM \
    --dataset Celeba  --trial_seed 1 --model_save --epochs 10 --stage 3
  ```
  
  - Train invariant CycleGAN (JS-divergence): 
  
  ```shell
  # pretrain
  
  # pretrain celeba
  CUDA_VISIBLE_DEVICES=2 python train_cyclegan.py --dataroot ./datasets/cmnist_0.6  \
  --model cycle_gan --pool_size 50 --no_dropout \
  --gan_mode vanilla  --eval --netG unet_128 --netD basic \
  --dataset Celeba  --batch_size 32 --obj vanilla --print_freq 500 --name vanilla_gender_bs32 --n_epochs 12 --image_size 128 --gender
  
  # pretrain male_nonblond & female_blond
  CUDA_VISIBLE_DEVICES=2 python train_cyclegan.py --dataroot ./datasets/cmnist_0.6  \
  --model cycle_gan --pool_size 50 --no_dropout \
  --gan_mode vanilla  --eval --netG unet_128 --netD basic \
  --dataset Celeba  --batch_size 32 --obj vanilla --print_freq 500 --name vanilla_bs32_2 --n_epochs 12 --image_size 128
  
  #gender
  
  CUDA_VISIBLE_DEVICES=2 python train_cyclegan.py --dataroot ./datasets/cmnist_0.6  \
  --model cycle_gan --pool_size 50 --no_dropout \
  --gan_mode vanilla  --eval --netG unet_128 --netD basic \
  --dataset Celeba  --batch_size 32 --obj js --print_freq 500 --name irm_y_g_2 --n_epochs 12 --image_size 128 --pretrain --alg IRM --gender
  
  CUDA_VISIBLE_DEVICES=3 python train_cyclegan.py --dataroot ./datasets/cmnist_0.6  \
  --model cycle_gan --pool_size 50 --no_dropout \
  --gan_mode vanilla  --eval --netG unet_128 --netD basic \
  --dataset Celeba  --batch_size 32 --obj js --print_freq 500 --name trm_y_g --n_epochs 12 --image_size 128 --pretrain --alg TRM --gender
  
  CUDA_VISIBLE_DEVICES=2 python train_cyclegan.py --dataroot ./datasets/cmnist_0.6  \
  --model cycle_gan --pool_size 50 --no_dropout \
  --gan_mode vanilla  --eval --netG unet_128 --netD basic \
  --dataset Celeba  --batch_size 32 --obj js --print_freq 500 --name trm_y_g_2 --image_size 128 --pretrain --alg TRM --gender --n_epochs 4
  
  
  CUDA_VISIBLE_DEVICES=1 python train_cyclegan.py --dataroot ./datasets/cmnist_0.6  \
  --model cycle_gan --pool_size 50 --no_dropout \
  --gan_mode vanilla  --eval --netG unet_128 --netD basic \
  --dataset Celeba  --batch_size 32 --obj js --print_freq 500 --name irm --n_epochs 10 --image_size 128 --pretrain --alg IRM
  
  CUDA_VISIBLE_DEVICES=2 python train_cyclegan.py --dataroot ./datasets/cmnist_0.6  \
  --model cycle_gan --pool_size 50 --no_dropout \
  --gan_mode vanilla  --eval --netG unet_128 --netD basic \
  --dataset Celeba  --batch_size 32 --obj js --print_freq 500 --name Celeba_mldg_2 --n_epochs 10 --image_size 128 --pretrain --alg MLDG 
  
  
  CUDA_VISIBLE_DEVICES=1 python train_cyclegan.py --dataroot ./datasets/cmnist_0.6  \
  --model cycle_gan --pool_size 50 --no_dropout \
  --gan_mode vanilla  --eval --netG unet_128 --netD basic \
  --dataset Celeba  --batch_size 32 --obj js --print_freq 500 --name mldg_y_g --n_epochs 12 --image_size 128 --pretrain --alg MLDG --gender
  
  CUDA_VISIBLE_DEVICES=3 python train_cyclegan.py --dataroot ./datasets/cmnist_0.6  \
  --model cycle_gan --pool_size 50 --no_dropout \
  --gan_mode vanilla  --eval --netG unet_128 --netD basic \
  --dataset Celeba  --batch_size 32 --obj js --print_freq 500 --name erm_g --n_epochs 12 --image_size 128 --pretrain --alg ERM --gender --oracle
  
  CUDA_VISIBLE_DEVICES=2 python train_cyclegan.py --dataroot ./datasets/cmnist_0.6  \
  --model cycle_gan --pool_size 50 --no_dropout \
  --gan_mode vanilla  --eval --netG unet_128 --netD basic \
  --dataset Celeba  --batch_size 32 --obj vanilla --print_freq 500 --name vanilla_bs32_2 --n_epochs 12 --image_size 128 --pretrain --alg ERM
  
  CUDA_VISIBLE_DEVICES=1 python train_cyclegan.py --dataroot ./datasets/cmnist_0.6  \
  --model cycle_gan --pool_size 50 --no_dropout \
  --gan_mode vanilla  --eval --netG unet_128 --netD basic \
  --dataset Celeba  --batch_size 32 --obj vanilla --print_freq 500 --name vanilla_bs32_test --n_epochs 12 --image_size 128 --alg ERM --gender
  ```
  
  - Evaluate FID score
  
    
  
    **Step 1: generate images**
  
    ```shell
    
    example:
    
    CUDA_VISIBLE_DEVICES=3 python3 generate.py --dataroot ./datasets/cmnist_0.6 --netG unet_128 --gan_mode vanilla --model cycle_gan --netD basic --dataset Celeba --name irm_y_g --image_size 128 --out_path irm_y_g --obj js --alg IRM --resume_epoch 2 --gender
    
    
    CUDA_VISIBLE_DEVICES=3 python3 generate.py --dataroot ./datasets/cmnist_0.6 --netG unet_128 --gan_mode vanilla --model cycle_gan --netD basic --dataset Celeba --name trm --image_size 128 --out_path celeba_js_trm_2 --obj js --alg TRM --resume_epoch 2
    
    CUDA_VISIBLE_DEVICES=3 python3 generate.py --dataroot ./datasets/cmnist_0.6 --netG unet_128 --gan_mode vanilla --model cycle_gan --netD basic --dataset Celeba --image_size 128 --obj js --out_path celeba_js_fish_6 --name fish --alg Fish --resume_epoch 6
    ```
  
    Visualization:
  
    ```
    CUDA_VISIBLE_DEVICES=3 python3 generate.py --dataroot ./datasets/cmnist_0.6 --netG unet_128 --gan_mode vanilla --model cycle_gan --netD basic --dataset Celeba --name trm --image_size 128 --out_path celeba_js_trm_2 --obj js --alg TRM --resume_epoch 2 --save
    
    CUDA_VISIBLE_DEVICES=2 python3 generate.py --dataroot ./datasets/cmnist_0.6 --netG unet_128 --gan_mode vanilla --model cycle_gan --netD basic --dataset Celeba --name vanilla_bs32 --image_size 128 --out_path vanilla --obj vanilla --alg ERM --resume_epoch 12 --save
    
    CUDA_VISIBLE_DEVICES=2 python3 generate.py --dataroot ./datasets/cmnist_0.6 --netG unet_128 --gan_mode vanilla --model cycle_gan --netD basic --dataset Celeba --name js_bs32_pretrain --image_size 128 --out_path celeba_js_12 --obj js --alg ERM --resume_epoch 4 --save
    
    CUDA_VISIBLE_DEVICES=2 python3 generate.py --dataroot ./datasets/cmnist_0.6 --netG unet_128 --gan_mode vanilla --model cycle_gan --netD basic --dataset Celeba --name Celeba_mldg --image_size 128 --out_path celeba_mldg_5 --obj js --alg MLDG --resume_epoch 5 --save
    
    CUDA_VISIBLE_DEVICES=2 python3 generate.py --dataroot ./datasets/cmnist_0.6 --netG unet_128 --gan_mode vanilla --model cycle_gan --netD basic --dataset Celeba --name erm_y_g --image_size 128 --out_path oracle_js_gender --obj js --alg ERM --resume_epoch 5 --save --gender
    
    
    CUDA_VISIBLE_DEVICES=2 python3 generate.py --dataroot ./datasets/cmnist_0.6 --netG unet_128 --gan_mode vanilla --model cycle_gan --netD basic --dataset Celeba --name vanilla_gender_bs32 --image_size 128 --out_path vanilla_gender --obj vanilla --alg ERM --resume_epoch 12 --save
    ```
  
    
  
    **Step 2: calculate FID score**
  
    ```shell
    python3 evaluation_tensorflow.py path --gpu 0 --celeba --name refine
    
    
    example:
    
    python3 evaluation_tensorflow.py ./generated_images/irm_y_g --gpu 0 --celeba --name refine
    
    CUDA_VISIBLE_DEVCEIS=3 python3 evaluation_tensorflow.py ./generated_images/vanilla --gpu 0 --celeba --name refine
    ```
  
  







