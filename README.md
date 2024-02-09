# Dynamic Time-aware Continual User Representation Learning

## Introduction
Implementation of **Dynamic Time-aware Continual User Representation Learning**.  
Continual user representation learning method in a practical scenario. The key idea is to transfer knowledge in both forward/backward directions to prevent catastrophic forgetting while allowing the previous knowledge to adapt to the current data distribution.


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

