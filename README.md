# SAGAN
ConvNet Final Project - Self-Attention Generative Adversarial Networks

* To run any experiment, you first need to download the following model 
for calculating the inception score, and copy it to the project folder:
https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth
(or instead you can run the commands from the path /home/wolf/sagieb/course/liorv/SAGAN, or copy the inception score model from there)

* I used the python environment of homework 4.

* Every 5 epochs (by default, can be set by 'model_save_epoch' argument) 
a checkpoint, scores graph and a log, are being saved in the default path './checkpoints'. 
This path can be changed by setting the argument 'model_save_path'.

* Every 1000 steps (by default, can be set by 'sample_save_step' argument) 
generated images are being saved in the default path './samples'.
This path can be changed by setting the argument 'sample_path'.


Here are the commands for reproducing the experiments:


cifar_baseline:
--------------
```
python train.py --data_path=/home/wolf/sagieb/course/liorv/SAGAN/data --model=baseline --version=cifar_baseline --dataset=cifar --num_epochs=30 --im_size=32 --g_ch=256 --d_ch=128 --model_save_epoch=5 --sample_save_step=500 --calc_score_step=500
```


cifar_sn_on_g_d:
---------------
```
python train.py --data_path=/home/wolf/sagieb/course/liorv/SAGAN/data --model=sn_on_g_d --version=cifar_sn_on_g_d --dataset=cifar --num_epochs=30 --im_size=32 --g_ch=256 --d_ch=128 --model_save_epoch=5 --sample_save_step=500 --calc_score_step=500
```


cifar_sn_on_g_d_ttur:
--------------------
```
python train.py --data_path=/home/wolf/sagieb/course/liorv/SAGAN/data --model=sn_on_g_d_ttur --version=cifar_sn_on_g_d_ttur --dataset=cifar --num_epochs=30 --im_size=32 --g_ch=256 --d_ch=128 --model_save_epoch=5 --sample_save_step=500 --calc_score_step=500
```


cifar_sagan_k8:
--------------
```
python train.py --data_path=/home/wolf/sagieb/course/liorv/SAGAN/data --model=sagan --version=cifar_sagan_k8 --dataset=cifar --num_epochs=30 --im_size=32 --g_ch=256 --d_ch=128 --feat_k=8 --model_save_epoch=5 --calc_score_step=500 --sample_save_step=500
```


cifar_sagan_k16:
--------------
```
python train.py --data_path=/home/wolf/sagieb/course/liorv/SAGAN/data --model=sagan --version=cifar_sagan_k16 --dataset=cifar --num_epochs=30 --im_size=32 --g_ch=256 --d_ch=128 --feat_k=16 --model_save_epoch=5 --calc_score_step=500 --sample_save_step=500
```

lsun_baseline:
-------------
```
python train.py --data_path=/home/wolf/sagieb/course/liorv/SAGAN/data --model=baseline --version=lsun_baseline --dataset=lsun --num_epochs=10 --im_size=64 --g_ch=64 --d_ch=64 --model_save_epoch=5 --calc_score_step=500 --sample_save_step=1000
```


lsun_sn_on_g_d:
--------------
```
python train.py --data_path=/home/wolf/sagieb/course/liorv/SAGAN/data --model=sn_on_g_d --version=lsun_sn_on_g_d --dataset=lsun --num_epochs=10 --im_size=64 --g_ch=64 --d_ch=64 --model_save_epoch=5 --calc_score_step=500 --sample_save_step=1000
```


lsun_sn_on_g_d_ttur:
-------------------
```
python train.py --data_path=/home/wolf/sagieb/course/liorv/SAGAN/data --model=sn_on_g_d_ttur --version=lsun_sn_on_g_d_ttur --dataset=lsun --num_epochs=10 --im_size=64 --g_ch=64 --d_ch=64 --model_save_epoch=5 --calc_score_step=500 --sample_save_step=1000
```


lsun_sagan_k8:
-------------
```
python train.py --data_path=/home/wolf/sagieb/course/liorv/SAGAN/data --model=sagan --version=lsun_sagan_k8 --dataset=lsun --num_epochs=10 --im_size=64 --g_ch=64 --d_ch=64 --feat_k=8 --model_save_epoch=5 --calc_score_step=500 --sample_save_step=1000
```


lsun_sagan_k16:
--------------
```
python train.py --data_path=/home/wolf/sagieb/course/liorv/SAGAN/data --model=sagan --version=lsun_sagan_k16 --dataset=lsun --num_epochs=10 --im_size=64 --g_ch=64 --d_ch=64 --feat_k=16 --model_save_epoch=5 --calc_score_step=500 --sample_save_step=1000
```


lsun_sagan_k32:
--------------
```
python train.py --data_path=/home/wolf/sagieb/course/liorv/SAGAN/data --model=sagan --version=lsun_sagan_k32 --dataset=lsun --num_epochs=30 --im_size=64 --g_ch=64 --d_ch=64 --feat_k=32 --model_save_epoch=5 --calc_score_step=500 --sample_save_step=1000
```

celeba_sagan_k32
----------------
```
python train.py --data_path=/home/wolf/sagieb/course/liorv/SAGAN/data --model=sagan --version=celeba_sagan_k32 --dataset=celeba --num_epochs=30 --im_size=64 --im_center_corp=160 --g_ch=64 --d_ch=64 --feat_k=32 --model_save_epoch=5 --calc_score_step=1000 --sample_save_step=2000
```