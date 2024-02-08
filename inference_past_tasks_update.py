import torch
import torch.nn as nn
import data_loader as data_loader
import ditto as generator_recsys

import math
import numpy as np
import argparse

import random
import time
from tqdm import tqdm
from tqdm import trange
import collections

from utils import *
import copy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_k', type=int, default=5,
                        help='Sample from top k predictions')
    parser.add_argument('--gpu_num', type=int, default=0,
                        help='Device (GPU) number')
    parser.add_argument('--epochs',type=int, default=10,
                        help='Total number of epochs')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='hyperpara-Adam')
    parser.add_argument('--alpha', default=0.7, type=float,
                        help='Controls the contribution of the knowledge retention')
    parser.add_argument('--savepath',type=str, default='./saved_models/past_task_inference_emb',
                        help='Save path of the results')
    parser.add_argument('--paths', type=str, default="./saved_models/task6.t7",
                        help='Load the model')
    parser.add_argument('--seed', type=int, default = 0,
                        help='Seed')
    parser.add_argument('--lr', type = float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--datapath', type=str, default="./Tmall_inference/task1_inference_click.csv",
                        help='data path want to inference')
    parser.add_argument('--datapath_index', type=str, default="Data/Tmall/index_task6.csv",
                        help='data path')
    parser.add_argument('--datapath_index_re', type=str, default='Data/Tmall/index_task_.csv',
                        help='item index dictionary path')
    parser.add_argument('--split_percentage', type=float, default=0.2,
                        help='0.2 means 80% training 20% testing')
    parser.add_argument('--smax',type=int, default = 50)
    parser.add_argument('--clipgrad',type=int, default = 1000)

    parser.add_argument('--batch', type=int,default=128)
    parser.add_argument('--model_name', type=str,default='NextitNet')
    parser.add_argument('--n_tasks', type=int, default = 6,
                        help='The total number of tasks that have been trained')
    parser.add_argument('--inference_task', type=int, default = 1,
                        help='The index of the task for which you want to perform inference')
    parser.add_argument('--n_neg', type=int, default=99)
    parser.add_argument('--rho', type=float, default=0.3)
    parser.add_argument('--classification',type=int, default= 0)

    args = parser.parse_args()
    
    
    """Set seed"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    """
    Load data
    "all_samples": consists of a sequence of item indices for each user and targets (labels).
    """
    if args.classification:
        dl = data_loader.Data_Loader_class({'model_type': 'generator', 'dir_name': args.datapath,'dir_name_index': args.datapath_index,'dir_name_index_re': args.datapath_index_re})
        items = dl.item_dict
        print("len(source)",len(items)) # The number of items.
        all_samples = dl.example
        targets_len_ = len(items)
        print('len(all_samples)',len(all_samples))
        
    elif args.inference_task !=1:
        dl = data_loader.Data_Loader_Sup({'model_type': 'generator', 'dir_name': args.datapath,'dir_name_index': args.datapath_index,'dir_name_index_re': args.datapath_index_re})
        items = dl.item_dict
        print("len(source)",len(items)) # The number of items.
        all_samples = dl.example
        targets_len_ = len(items)
        print('len(all_samples)',len(all_samples))
    else:
        dl = data_loader.Data_Loader_Inference({'model_type': 'generator', 'dir_name': args.datapath,'dir_name_index': args.datapath_index,'dir_name_index_re': args.datapath_index_re,'lambdafm_rho':args.rho})
        all_samples = dl.item
        items = dl.item_dict
        targets_len_ = len(items)
    # Shuffle the data.
    shuffle_indices = np.random.permutation(np.arange(len(all_samples)))
    all_samples = all_samples[shuffle_indices]

    #Split the data into train, validation, and test datasets.
    dev_sample_index = -1 * int(args.split_percentage * float(len(all_samples)))
    dev_sample_index_valid = int(dev_sample_index*0.75)
    train_set, valid_set_, test_set_ = all_samples[:dev_sample_index], all_samples[dev_sample_index:dev_sample_index_valid], all_samples[dev_sample_index_valid:]

    # Set GPU
    if args.gpu_num == 'cpu':
        args.device = 'cpu'
    else:
        args.device = torch.device("cuda:" + str(args.gpu_num) if torch.cuda.is_available() else "cpu")

    """
    Load information of previous tasks.
    "target_size": The list of the total number of past tasks' target.
    "task_dict": Model parameters of previous task.
    "current_task_dict": Current model parameters (before train current model). 
    """
    target_size, task_dict = inference_model(args)

    if args.inference_task == 1:
        item_size = torch.max(torch.LongTensor(test_set_[:,:-1]).to(args.device)).item()+1
    else:
        item_size = task_dict['embeding.weight'].shape[0]


    """
    Set model parameters
    "item_size": the total number of unique items.
    "dilated_channels": Dimension of item embedding and hidden state.
    "target_size": The list of the total number of past tasks' target.
    "dilations": dilation of convolutional layers.
    "kernel_size": kernel size of convolutional layers.
    "learning_rate": model learning rate.
    "batch_size": training batch size.
    "task_embs": The upper bound and lower bound of the initialization distribution for the task embedding.
    "num_task": The total number of tasks.
    """
    model_para = {
        #'item_size': task_dict['embeding.weight'].shape[0],
        'item_size' : item_size,
        'dilated_channels': 256,
        'target_item_size': target_size,
        'dilations': [1,4,1,4,1,4,1,4,],
        'kernel_size': 3,
        'learning_rate':args.lr,
        'batch_size':args.batch,
        'task_embs':[0,2],
        'num_task':args.n_tasks,
    }

    # Generate current model
    model = generator_recsys.NextItNet_Decoder(model_para).to(args.device)

    weight = model.state_dict()
    weight['embeding.weight'][:task_dict['embeding.weight'].shape[0],:] = task_dict['embeding.weight']
    task_dict['embeding.weight'] = weight['embeding.weight']
    del weight
    
    # Load the model parameters of the previous task into the current model.
    model.load_state_dict(task_dict, strict=False)
    for n,p in model.named_parameters():
            p.requires_grad = False
    
    # Inference the model
    model.eval()
    smax = args.smax
    list__ = [[] for i in range(6)]
    if not args.classification: # If the number of targets are large, calculate metrics such as MRR, Hit, NDCG else calcuate accuracy only.
        model_test(model_para, test_set_, model,0,args,list__,smax, backward=False,task_num = args.inference_task-1,new_task=args.n_tasks-1,len_items = task_dict['embeding.weight'].shape[0])
    else:
        model_test_acc(model_para, test_set_, model,0,args,list__,smax, backward=False,task_num = args.inference_task-1, new_task=args.n_tasks-1)

if __name__ == '__main__':
    curr_preds_5 = []
    rec_preds_5 = []
    ndcg_preds_5 = []
    curr_preds_20 = []
    rec_preds_20 = []
    ndcg_preds_20 = []
    main()

