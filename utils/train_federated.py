import logging
import os
import time
import warnings
import torch
from torch import nn
from torch import optim
import models
import numpy as np
from models.CNN_2d import CNN

def iid(dataset, num_users):
    '''
    将数据集划分为独立同分布
    '''
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def non_iid(dataset,num_users):
    '''
    将数据集划分为非独立同分布
    '''
    pass

def get_dataset(args):
    '''
    加载数据集
    '''
    train_dataset, test_dataset = Dataset(args,args.data_dir,args.normlizetype).data_preprare()
    if args.iid:
        user_groups=iid(train_dataset,args.num_users)
    else:
        user_groups=non_iid(train_dataset,args.num_users)
    return train_dataset, test_dataset, user_groups

#返回权重的平均值，即执行联邦平均算法
def average_weights(w):
    #w是经过多轮本地训练后统计的权重list，在参数默认的情况下，是一个长度为10的列表
    # 而每个元素都是一个字典，每个字典都包含了模型参数的名称（比如layer_input.weight或者layer_hidden.bias），以及其权重具体的值
    """
    Returns the average of the weights.
    """
    #深度复制，被复制的对象不会随着复制的对象的改变而改变，这里复制了第一个用户的权重字典。
    #随后，对于每一类参数进行循环，累加每个用户模型里对应参数的值，最后取平均获得平均后的模型。
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


class train_federated(object):
    def __init__(self, args, save_dir):
        self.args = args
        self.save_dir = save_dir

    def Setup(self):
        args=self.args

        #gpu
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))

        # Load the datasets
        if args.processing_type == 'O_A':
            from CNN_Datasets.O_A import datasets
            Dataset = getattr(datasets, args.data_name)
        elif args.processing_type == 'R_A':
            from CNN_Datasets.R_A import datasets
            Dataset = getattr(datasets, args.data_name)
        elif args.processing_type == 'R_NA':
            from CNN_Datasets.R_NA import datasets
            Dataset = getattr(datasets, args.data_name)
        else:
            raise Exception("processing type not implement")

        train_dataset, test_dataset, user_groups = get_dataset(args)

        # Define the model
        if args.model_name == 'cnn_2d':
            global_model = CNN()
        else:
            exit('Error: unrecognized model')

    def train(self):
        args=self.args
        # Set the model to train and send it to device.
        global_model.to(self.device)
        global_model.train()
        print(global_model)

        # copy weights
        global_weights = global_model.state_dict()

        # Training
        train_loss, train_accuracy = [], []
        val_acc_list, net_list = [], []
        cv_loss, cv_acc = [], []
        print_every = 2
        val_loss_pre, counter = 0, 0

        for epoch in tqdm(range(args.epochs)):
            local_weights, local_losses = [], []
            print(f'\n | Global Training Round : {epoch+1} |\n')

            global_model.train()
            m = max(int(args.frac * args.num_users), 1)#随机选比例为frac的用户
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            #本地训练
            '''
            LocalUpdate还没写
            '''