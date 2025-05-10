# [SIGIR'25] Dynamic Time-aware Continual User Representation Learning

## Introduction
Implementation of **Dynamic Time-aware Continual User Representation Learning**.  
Continual user representation learning method in a practical scenario. The key idea is to transfer knowledge in both forward/backward directions to prevent catastrophic forgetting while allowing the previous knowledge to adapt to the current item distribution.


## Basics
1. The train code for $T_1$ is in `train_task1.py`.
2. The main train code for $T_{2:i}$ is in `train_ditto.py`.
3. The inference code is in `inference_past_tasks_update.py`.


## Dataset
You can download the datasets (Tmall, Movielens, and Taobao) from the following links.
1. Tmall
  https://tianchi.aliyun.com/dataset/42
2. ML
  https://grouplens.org/datasets/movielens/
3. Taobao
  https://tianchi.aliyun.com/dataset/649


## Requirments
- Pytorch version: 1.8.0
- Numpy version: 1.21.5


## Arguments
- `--datapath:` Path of the dataset.<br>
	- usage example :`--dataset ./Tmall/task1_click.csv`
- `--paths:` Path of the model trained on a previous task.<br>
	- usage example :`--paths ./saved_models/task1.t7`
- `--savepath:` Path to which the current model is saved.<br>
	- usage example : `--savepath ./saved_models/task2`
- `--n_tasks:`  Total number of the tasks.<br>
	- usage example :`--n_tasks 2`
- `--datapath_index:` Path of the item index dictionary (i.e., `Data/Session/index.csv`).<br>
	- usage example :`--datapath_index Data/Session/index.csv`
	- Note that the file `index.csv` is automatically generated when running Task 1.
Specifically, when training the model on Task 1, `data_loader` generates the `index.csv` file, which contains the index information for all items in Task 1.<br>
- `--lr:` Learning rate.<br>
	- usage example : `--lr 0.0001`
- `--alpha:` A hyperparameter that controls the contribution of the forward knowledge transfer.<br>
	- usage example : `--alpha 0.7`
- `--beta:` A hyperparameter that controls the contribution of the backward knowledge transfer for Task 1.<br>
	- usage example : `--beta 0.4`
- `--gamma:` A hyperparameter that controls the contribution of the backward knowledge transfer for rest tasks.<br>
	- usage example : `--gamma 0.8`
- `--smax:` Positive scaling hyper-parameter.<br>
	- usage example : `--smax 50`



## Source code of the backbone network
- The source code of the backbone network is referenced from:
  - https://github.com/yuangh-x/2022-NIPS-Tenrec
  - https://github.com/syiswell/NextItNet-Pytorch
  - https://github.com/Sein-Kim/TERACON
