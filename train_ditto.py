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
from torchmetrics.functional import pairwise_cosine_similarity

from utils import *
import copy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_k', type=int, default=5,
                        help='Sample from top k predictions')
    parser.add_argument('--gpu_num', type=int, default=8,
                        help='Device (GPU) number')
    parser.add_argument('--epochs',type=int, default=100,
                        help='Total number of epochs')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='hyperpara-Adam')
    parser.add_argument('--alpha', default=0.7, type=float,
                        help='Controls the contribution of the forward knowledge transfer')
    parser.add_argument('--beta', default=0.4, type=float, 
                        help='Controls the contribution of the backward knowledge transfer (task1)')
    parser.add_argument('--gamma', default=0.8, type=float,
                        help='Controls the contribution of the backward knowledge transfer (task2~)')
    parser.add_argument('--savepath',type=str, default='./saved_models/task3',
                        help='Save path of current model')
    parser.add_argument('--paths', type=str, default='./saved_models/task2.t7',
                        help='Load path of past model')
    parser.add_argument('--seed', type=int, default = 0,
                        help='Seed')
    parser.add_argument('--lr', type = float, default=0.0005,
                        help='Learning rate')
    parser.add_argument('--datapath', type=str, default='./Tmall/task3_purchase.csv',
                        help='data path')
    parser.add_argument('--datapath_index', type=str, default='Data/Tmall/index_task2.csv',
                        help='item index dictionary path')
    parser.add_argument('--datapath_index_re', type=str, default='Data/Tmall/index_task3.csv',
                        help='item index dictionary path')
    parser.add_argument('--split_percentage', type=float, default=0.2,
                        help='0.2 means 80% training 20% testing')
    parser.add_argument('--smax',type=int, default = 50)
    parser.add_argument('--clipgrad',type=int, default = 1000)

    parser.add_argument('--batch', type=int,default=512)
    parser.add_argument('--model_name', type=str,default='NextitNet')
    parser.add_argument('--n_tasks', type=int, default = 3,
                        help='The total number of tasks')
    parser.add_argument('--n_neg', type=int, default=99)
    parser.add_argument('--classification',type=int, default= 0)
    parser.add_argument('--classification_task',type=str, default='5,6')
    parser.add_argument('--mean_embedding', type=str, default='')
    parser.add_argument('--task1_bkt', type=int, default=1)
    parser.add_argument('--rest_bkt', type=int, default=1)
    parser.add_argument('--fkt', type=int, default=1)

    args = parser.parse_args()
    
    cl_task = args.classification_task.split(',')
    args.classification_task = [int(s)-1 for s in cl_task]
    
    """Set seed"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    """
    Load data
    "all_samples": consists of a sequence of item indices for each user and targets (labels).
    """
    if args.classification:
        dl =  data_loader.Data_Loader_class({'model_type': 'generator', 'dir_name': args.datapath,'dir_name_index': args.datapath_index,'dir_name_index_re': args.datapath_index_re})
    else:
        dl = data_loader.Data_Loader_Sup({'model_type': 'generator', 'dir_name': args.datapath,'dir_name_index': args.datapath_index,'dir_name_index_re': args.datapath_index_re})
    items = dl.item_dict
    print("len(source)",len(items)) # The number of items.
    all_samples = dl.example
    print('len(all_samples)',len(all_samples))

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
    target_size, task_dict, current_task_dict = task_model(args)
    if args.classification:
        target_size.append(len(dl.target_dict))
    else:
        target_size.append(256) # Append the number of current task's target to list.


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
        'item_size': len(items),
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
    
    # Set the total number of tasks to generate previous model
    model_para['num_task'] = args.n_tasks - 1
    weight = model.state_dict()
    weight['embeding.weight'][:current_task_dict['embeding.weight'].shape[0],:] = current_task_dict['embeding.weight']
    current_task_dict['embeding.weight'] = weight['embeding.weight']
    del weight
    
    past_target_list = []
    for i in range(1, args.n_tasks):
        temp = torch.load("./saved_models/task%d_abl6.t7"%(i), map_location = torch.device(args.device))
        past_target_list.append(temp['net']['embeding.weight'].size(0))

    past_target_len = task_dict['embeding.weight'].shape[0]
    model_para = {
        'item_size': task_dict['embeding.weight'].shape[0],
        'dilated_channels': 256,
        'target_item_size': target_size,
        'dilations': [1,4,1,4,1,4,1,4,],
        'kernel_size': 3,
        'learning_rate':args.lr,
        'batch_size':args.batch,
        'task_embs':[0,2],
        'num_task':args.n_tasks-1,
    }
    model_para['num_task'] = args.n_tasks - 1
    # Generate past tasks model
    past_models =  generator_recsys.NextItNet_Decoder(model_para).to(args.device)
    past_models.load_state_dict(task_dict,strict=False) # Load model parameters of previous task.
    for n,p in past_models.named_parameters():
            p.requires_grad = False
    mean_emb = torch.mean(past_models.embeding.weight,axis=0)
    torch.save(mean_emb, './saved_models/mean_embedding_%s%s.t7' %(args.n_tasks-1, args.mean_embedding))
    
    # Load the model parameters of the previous task into the current model.
    model.load_state_dict(current_task_dict, strict=False)
    
    #Set loss function (criterion, criterion_)
    criterion = nn.CrossEntropyLoss() # Classification Loss
    criterion_ = nn.MSELoss() # Knowledge transfer loss

    #Set optimizer (Adam)
    optimizer = torch.optim.Adam(model.parameters(), lr=model_para['learning_rate'], weight_decay=0)
    
    # Set early stop
    count = 0
    best_acc = 0

    # Initialize the sampling ratio.
    random_batch_size = [0.9 for _ in range(model_para['num_task'])]

    past_models.embeding.weight = torch.nn.Parameter(torch.cat((past_models.embeding.weight, model.embeding.weight[past_models.embeding.weight.size(0):,:]),dim=0))

    #Train
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        transfer_loss_t=0
        bkt_loss_t1 = 0
        bkt_loss_t2 = 0
        correct = 0
        total = 0
        batch_size = model_para['batch_size']
        batch_num = train_set.shape[0] / batch_size
        start = time.time()
        INFO_LOG("-------------------------------------------------------train")
        for batch_idx, batch_sam in enumerate(getBatch(train_set, batch_size)):
            #Annealing trick.
            smax = args.smax
            r = batch_num
            s = (smax-1/smax)*batch_idx/r+1/smax
            optimizer.zero_grad()
            
            
            """
            Split the inputs and targets (labels).
            The target value is located at the last index of each data line.
            """
            inputs = torch.LongTensor(batch_sam[:, :-1]).to(args.device)
            target = torch.LongTensor(batch_sam[:,-1]).to(args.device)


            inputs_masking = (inputs>(past_target_len-1)).sum(axis=1) == False

            mask_list = []
            for i in range(model_para['num_task']):
                mask_list.append((inputs>(past_target_list[i])).sum(axis=1)==False)
            

            random_buffer = [[i for i in range(len(inputs))] for n in range(model_para['num_task'])]
            random_buffer = [torch.tensor(buf).to(args.device) for buf in random_buffer]
            
            outputs,current_masks,_ = model(inputs,s,[], args.n_tasks-1, args.gpu_num,onecall=True,new_task=(args.n_tasks-1))

            exist_idx = []
            new_idx = []
            for i in range(model_para['num_task']):
                mean_emb = torch.load('./saved_models/mean_embedding_%s%s.t7'%(i+1, args.mean_embedding), map_location=torch.device(args.device))
                _,prev_m, prev_o = past_models(inputs,s,[], i,args.gpu_num,onecall=True,new_task=(args.n_tasks-1))

                exist_sim = pairwise_cosine_similarity(mean_emb.reshape(1,-1),prev_o)
                v,idx = torch.topk(exist_sim[0], k=min(prev_o.size(0), max(int(prev_o.size(0)*random_batch_size[i]),20)))
                target_idx = np.array(idx.cpu())
                exist_idx.append(target_idx)

            for i in range(model_para['num_task']):
                mean_emb = torch.load('./saved_models/mean_embedding_%s%s.t7'%(i+1, args.mean_embedding), map_location=torch.device(args.device))
                _,prev_m,prev_o = past_models(inputs,s,[], i,args.gpu_num,onecall=True,new_task=(args.n_tasks-1))

                new_sim = pairwise_cosine_similarity(mean_emb.reshape(1,-1),prev_o)
                new_sim = (-1)*new_sim
                v,idx = torch.topk(new_sim[0], k=min(prev_o.size(0), max(int(prev_o.size(0)*random_batch_size[i]),20)))
                target_idx = np.array(idx.cpu())
                new_idx.append(target_idx)



            ########Classification##########
            if not args.classification:

                neg_target = np.random.choice(len(items), target.shape[0]*args.n_neg)
                neg_target = torch.LongTensor(neg_target).to(args.device)

                target_emb_positive = model.embeding(target)
                target_emb_negative = model.embeding(neg_target)
                        
            
            prev_outputs, prev_masks = [], []
            for i in range(model_para['num_task']):
                if i ==0:
                    _,prev_m,prev_o = past_models(inputs[exist_idx[i],:-1][inputs_masking[exist_idx[i]],:],s,[], i,args.gpu_num,onecall=True,new_task=(args.n_tasks-1))
                else:
                    _,prev_m,prev_o = past_models(inputs[exist_idx[i],:][inputs_masking[exist_idx[i]],:],s,[], i,args.gpu_num,onecall=True,new_task=(args.n_tasks-1))
                prev_outputs.append(prev_o)
                prev_masks.append(prev_m)

            neg_target_pseudo = [torch.LongTensor(np.random.choice(past_target_len, inputs_masking[r_idx].sum().item()*args.n_neg)).to(args.device) for r_idx in exist_idx]

            # Get outputs of newly emerged task using current model.
            # Newly emerged task masks are stored in "current_masks" list.
            # We substract 1 at args.n_tasks (i.e., new_task = args.n_tasks-1) Because, the index of tasks start from 0.
            
            outputs,current_masks,_ = model(inputs,s,[], args.n_tasks-1, args.gpu_num,onecall=True,new_task=(args.n_tasks-1))

            if not args.classification:
                outputs_dup = outputs.unsqueeze(1).repeat(1,args.n_neg,1).view(-1,outputs.size(1))
                pos_logits = torch.sum((outputs*target_emb_positive), dim=-1)
                pos_logits = pos_logits.unsqueeze(1).repeat(1,args.n_neg,1).squeeze(2)
                pos_logits = pos_logits.squeeze(0)
                neg_logits = torch.sum((outputs_dup*target_emb_negative),dim=-1)
                loss = -(pos_logits-neg_logits).sigmoid().log().sum() + 1e-8

            else:
                loss = criterion(outputs, target)


            ########Forward Knowledge Transfer##########
            # Get output of previous tasks (i.e., pseudo label predictions) using current model.
            # The pseudo label predictions are stored in "student_outpus" list.
            student_outputs, _masks = [], []
            for i in range(model_para['num_task']):
                if i ==0:
            # We substract 1 at args.n_tasks (i.e., new_task = args.n_tasks-1) Because, the index of tasks start from 0.
                    _,m,student_o = model(inputs[exist_idx[i],:-1][inputs_masking[exist_idx[i]],:],s,[], i, args.gpu_num,onecall=True,backward=True,new_task=(args.n_tasks-1))
                else:
            # We substract 1 at args.n_tasks (i.e., new_task = args.n_tasks-1) Because, the index of tasks start from 0.
                    _,m,student_o = model(inputs[exist_idx[i],:][inputs_masking[exist_idx[i]],:],s,[], i, args.gpu_num,onecall=True,backward=True,new_task=(args.n_tasks-1))
                student_outputs.append(student_o)
            
            #Calculate FKT Loss.
            transfer_loss = torch.tensor(0, dtype=torch.float32).to(args.device)
            if args.fkt == 1:
                for i in range(len(student_outputs)):
                    if not (i in args.classification_task):
                        target_emb_negative = model.embeding(neg_target_pseudo[i]).data
                        studnet_outdup = student_outputs[i].unsqueeze(1).repeat(1,args.n_neg,1).view(-1,outputs.size(1))
                        prev_outdup = prev_outputs[i].unsqueeze(1).repeat(1,args.n_neg,1).view(-1,outputs.size(1))
                        
                        student_neg_logits = torch.sum((studnet_outdup*target_emb_negative),dim=-1)
                        prev_neg_logits = torch.sum((prev_outdup*target_emb_negative),dim=-1)
                        transfer_loss += random_batch_size[i]/sum(random_batch_size)*criterion_(student_neg_logits,prev_neg_logits)+ 1e-6
                    else:
                        transfer_loss += random_batch_size[i]/sum(random_batch_size)*criterion_(student_outputs[i],prev_outputs[i])
            
                    
            
            ########Backward Knowledge Transfer##########
            bkt_loss1 = torch.tensor(0, dtype=torch.float32).to(args.device)
            bkt_loss2 = torch.tensor(0, dtype=torch.float32).to(args.device)
            for i in range(model_para['num_task']):
                if i == 0 and args.task1_bkt == 1:
                    count1 = []
                    count2 = []
                    for j in range(inputs[~inputs_masking].size(0)):
                        temp_mask = (inputs[~inputs_masking][j]>past_target_len-1) == True
                        count1.append(torch.unique(inputs[~inputs_masking][j][temp_mask]))
                        count2.append(len(count1[j]))

                    v, select = torch.topk(torch.tensor(count2).to(args.device), min(int(len(count2)), max(int(len(count2)*random_batch_size[i]),20)))

                
                    tar = inputs[~inputs_masking,:][select,1:-1].view([-1])
                    neg_tar = np.random.choice(len(items), tar.shape[0]*args.n_neg)
                    neg_tar = torch.LongTensor(neg_tar).to(args.device)
                    tar_emb_pos = model.embeding(tar)
                    tar_emb_neg = model.embeding(neg_tar)

                    
                    out, m,_ = model(inputs[~inputs_masking,:][select,:-2],s,[], i, args.gpu_num, onecall=False, backward=True, new_task=(args.n_tasks-1))

                    outputs_dup = out.unsqueeze(1).repeat(1,args.n_neg,1).view(-1,out.size(1))
                    pos_logits = torch.sum((out*tar_emb_pos), dim=-1)
                    pos_logits = pos_logits.unsqueeze(1).repeat(1,args.n_neg,1).squeeze(2)
                    pos_logits = pos_logits.squeeze(0)
                    neg_logits = torch.sum((outputs_dup*tar_emb_neg),dim=-1)
                    bkt_loss1 = bkt_loss1 -(pos_logits-neg_logits).sigmoid().log().sum()

                elif not (i in args.classification_task) and args.rest_bkt == 1 and i != 0:
                    inp = inputs[new_idx[i],:][~mask_list[i][new_idx[i]],:]
                    if inp.size(0) != 0:
                        aug_inp1 = aug_masking(inp, 0.2, args)
                        aug_inp2 = aug_substitute(inp, 0.2, args, model)
                        aug_seq = torch.cat((aug_inp1, aug_inp2), dim=0)
                        _, m, out = model(aug_seq,s,[],i, args.gpu_num, onecall=True, backward=True, new_task=(args.n_tasks-1))
                        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                        for idx in range(out.size(0)):
                            own = torch.stack([out[idx]]*out.size(0), dim=0)
                            simm = cos(own, out)
                            if idx < int(out.size(0)/2):
                                bkt_loss2 += -(torch.exp(simm[idx+int(out.size(0)/2)])/torch.sum(torch.exp(simm[torch.arange(out.size(0))!=idx]))).log()
                            else:
                                bkt_loss2 += -(torch.exp(simm[idx-int(out.size(0)/2)])/torch.sum(torch.exp(simm[torch.arange(out.size(0))!= idx]))).log()
                        
            

            # Calculate sampling ratio for each tasks..
            random_batch_size = [sample_ratio(prev_masks[i],current_masks) for i in range(len(student_outputs))]

            clipgrad = args.clipgrad

            if not torch.isfinite(loss):
                print("Occured Nan", loss)
                loss = 0
                total += 0
            else:
                loss += (args.alpha * transfer_loss) + (args.beta * bkt_loss1) + (args.gamma * bkt_loss2)
                loss.backward()
                train_loss += loss.item()
                transfer_loss_t +=args.alpha * transfer_loss.item()
                bkt_loss_t1 += args.beta * bkt_loss1.item()
                bkt_loss_t2 += args.gamma * bkt_loss2.item()
                # Use gradient clipping if needed
                thres_cosh = 50
                # for n,p in model.named_parameters():
                #     if ('.fec' in n):
                #         num=torch.cosh(torch.clamp(s*p.data,-thres_cosh,thres_cosh))+1
                #         den=torch.cosh(p.data)+1
                #         if p.grad != None:
                #             p.grad.data*=smax/s*num/den
                # torch.nn.utils.clip_grad_norm_(model.parameters(),clipgrad)
                optimizer.step()

            thres_emb = 6
            for n,p in model.named_parameters():
                if ('.ec' in n):
                    p.data=torch.clamp(p.data,-thres_emb,thres_emb)
                    
            # _,predicted = outputs.max(1)
            # total +=target.size(0)
            # correct += predicted.eq(target).sum().item()
            
            # We display the training results for each epoch, printing them out 100 times.
            if batch_idx % max(10, batch_num//100) == 0:
                INFO_LOG("epoch: {}\t {}/{}".format(epoch, batch_idx, batch_num))
                print('Transfer Loss: %.3f'%(transfer_loss_t/(batch_idx+1)))
                print('BKT Loss1: %.3f'%(bkt_loss_t1/(batch_idx+1)))
                print('BKT Loss2: %.3f'%(bkt_loss_t2/(batch_idx+1)))
                print('Loss: %.3f' % (
                    train_loss / (batch_idx + 1)))
        end = time.time()

        model.eval()
        correct = 0
        total = 0
        batch_size_test = model_para['batch_size'] #Batch size of validation and test
        batch_num_test = valid_set_.shape[0] / batch_size
        list_ = [[] for i in range(6)]
        INFO_LOG("-------------------------------------------------------valid")
        with torch.no_grad():
            start = time.time()
            for batch_idx, batch_sam in enumerate(getBatch(valid_set_, batch_size_test)):
                inputs = torch.LongTensor(batch_sam[:, :-1]).to(args.device)
                target = torch.LongTensor(batch_sam[:,-1]).to(args.device)

                # We substract 1 at args.n_tasks (i.e., new_task = args.n_tasks-1) Because, the index of tasks start from 0.
                outputs, masks1,_ = model(inputs,smax,[], args.n_tasks-1,args.gpu_num,onecall=True,new_task=(args.n_tasks-1))

                neg_target = np.random.choice(len(items), target.shape[0]*args.n_neg)
                neg_target = torch.LongTensor(neg_target).to(args.device)
                target_emb_positive = model.embeding(target)
                target_emb_negative = model.embeding(neg_target)

                outputs_dup = outputs.unsqueeze(1).repeat(1,args.n_neg,1).view(-1,outputs.size(1))
                pos_logits = torch.sum((outputs*target_emb_positive), dim=-1).reshape(-1,1)
                neg_logits = torch.sum((outputs_dup*target_emb_negative),dim=-1).reshape(pos_logits.shape[0],-1)

                outputs_ = torch.cat((pos_logits,neg_logits),axis=1)

                list_toy = [[] for i in range(6)]
                
                if not args.classification:
                    _, sort_idx_20 = torch.topk(outputs_, k=args.top_k + 15, sorted=True)
                    _, sort_idx_5 = torch.topk(outputs_, k=args.top_k, sorted=True)
                    tar = torch.zeros(pos_logits.shape[0])

                    result_ = accuracy_test(sort_idx_5.data.cpu().numpy(), sort_idx_20.data.cpu().numpy(), tar.data.cpu().numpy(),
                            batch_idx, batch_num_test, epoch, args, list_toy)
                    for i in range(len(list_)):
                        list_[i] +=result_[i]
                else:
                    _, predicted = outputs.max(1)
                    total += target.size(0)
                    correct+= predicted.eq(target).sum().item()
            end = time.time()
        if not args.classification: # If the number of targets are large, calculate metrics such as MRR, Hit, NDCG else calcuate accuracy only.
            INFO_LOG("Accuracy mrr_5: {}".format(sum(list_[0]) / float(len(list_[0]))))
            INFO_LOG("Accuracy mrr_20: {}".format(sum(list_[3]) / float(len(list_[3]))))
            INFO_LOG("Accuracy hit_5: {}".format(sum(list_[1]) / float(len(list_[1]))))
            INFO_LOG("Accuracy hit_20: {}".format(sum(list_[4]) / float(len(list_[4]))))
            INFO_LOG("Accuracy ndcg_5: {}".format(sum(list_[2]) / float(len(list_[2]))))
            INFO_LOG("Accuracy ndcg_20: {}".format(sum(list_[5]) / float(len(list_[5]))))
            acc = sum(result_[0]) / float(len(result_[0]))
        else:
            print('Acc(hit@1): %.3f%% (%d/%d)' % (100. * correct / total, correct, total))
            acc = 100.*correct/total

        # Conduct test at best validation (store the model parameters)
        # The last recorded test score is considered as the test score at best validation score, as the test is performed whenever a new validation score is updated.
        if epoch == 0:
            best_acc =acc
            count = 0
            print('-----testing in best validation-----')
            list__ = [[] for i in range(6)]
            # We substract 1 at args.n_tasks (i.e., new_task = args.n_tasks-1) Because, the index of tasks start from 0.
            if not args.classification: # If the number of targets are large, calculate metrics such as MRR, Hit, NDCG else calcuate accuracy only.
                model_test(model_para, test_set_, model,epoch,args,list__,smax, backward=False,task_num = args.n_tasks-1,new_task=args.n_tasks-1,len_items=len(items))
            else:
                model_test_acc(model_para, test_set_, model,epoch,args,list__,smax, backward=False,task_num = args.n_tasks-1, new_task=args.n_tasks-1)

        else:
            if best_acc < acc:
                best_acc = acc
                count = 0
                print('-----testing in best validation-----')
                list__ = [[] for i in range(6)]
                # We substract 1 at args.n_tasks (i.e., new_task = args.n_tasks-1) Because, the index of tasks start from 0.
                if not args.classification: # If the number of targets are large, calculate metrics such as MRR, Hit, NDCG else calcuate accuracy only.
                    model_test(model_para, test_set_, model,epoch,args,list__,smax, backward=False,task_num = args.n_tasks-1,new_task=args.n_tasks-1,len_items = len(items))
                else:
                    model_test_acc(model_para, test_set_, model,epoch,args,list__,smax, backward=False,task_num = args.n_tasks-1, new_task=args.n_tasks-1)
            else: # Early stop
                count+=1

        if count == 2: # Early stop
            break
        print('count', count)
        INFO_LOG("TIME FOR EPOCH During Training: {}".format(end - start))
        INFO_LOG("TIME FOR BATCH (mins): {}".format((end - start) / batch_num))


if __name__ == '__main__':
    curr_preds_5 = []
    rec_preds_5 = []
    ndcg_preds_5 = []
    curr_preds_20 = []
    rec_preds_20 = []
    ndcg_preds_20 = []
    main()

