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
from utils.Update import LocalUpdate,test_inference
from tqdm import tqdm
import copy
from utils.train_federated import get_dataset

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


class train_federatedPlus(object):
    def __init__(self, args, save_dir):
        self.args = args
        self.save_dir = save_dir

    def train(self):
        start_time = time.time()
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

        # Define the model
        if args.model_name == 'cnn_2d':
            if args.data_name == 'CWRUSTFT':
                global_model = CNN()
            elif args.data_name == 'PUSTFT':
                global_model =CNN(out_channel=3)
        else:
            exit('Error: unrecognized model')
        #数据集
        train_dataset, test_dataset, user_groups = get_dataset(args)
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
        print_every = 1
        val_loss_pre, counter = 0, 0

        for epoch in tqdm(range(args.epochs)):
            local_weights, local_losses = [], []
            logging.info(f'\n | Global Training Round : {epoch+1} |\n')

            global_model.train()
            m = max(int(args.frac * args.num_users), 1)#随机选比例为frac的用户
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            #本地训练
            '''
            LocalUpdate还没写
            '''
            for idx in idxs_users:
                local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], i=idx)
                w,loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))

            global_weights = average_weights(local_weights)

            global_model.load_state_dict(global_weights)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            #每轮训练计算所有用户的平均训练精度
            list_acc, list_loss = [], []
            global_model.eval()
            for c in range(args.num_users):
                local_model = LocalUpdate(args=args, dataset=train_dataset,idxs=user_groups[idx],i=c)
                acc, loss = local_model.inference(model=global_model)
                list_acc.append(acc)
                list_loss.append(loss)
            train_accuracy.append(sum(list_acc)/len(list_acc))

            # print global training loss after every 'i' rounds 打印全局loss
            if (epoch+1) % print_every == 0:
                logging.info(f' \nAvg Training Stats after {epoch+1} global rounds:')
                logging.info(f'Training Loss : {np.mean(np.array(train_loss))}')
                logging.info('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

        #Test
        #联邦训练后，测试模型在测试集上的表现
        test_acc, test_loss = test_inference(args, global_model, test_dataset)

        logging.info(f' \n Results after {args.epochs} global rounds of training:')
        logging.info("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
        logging.info("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
        logging.info('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

        final_model=global_model.state_dict()
        torch.save(final_model, os.path.join(self.save_dir,
                                             'fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_final_model.pth'.
                                             format(args.data_name, args.model_name, args.epochs, args.frac,
                                                    args.iid, args.local_ep, args.local_bs)))

        # PLOTTING (optional)
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use('Agg')

        #Plot Loss curve
        plt.figure()
        plt.title('Training Loss vs Communication rounds')
        plt.plot(range(len(train_loss)), train_loss, color='r')
        plt.ylabel('Training loss')
        plt.xlabel('Communication Rounds')
        plt.savefig('{}/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
                    format(self.save_dir,args.data_name, args.model_name, args.epochs, args.frac,
                           args.iid, args.local_ep, args.local_bs))

        # Plot Average Accuracy vs Communication rounds
        plt.figure()
        plt.title('Average Accuracy vs Communication rounds')
        plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
        plt.ylabel('Average Accuracy')
        plt.xlabel('Communication Rounds')
        plt.savefig('{}/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
                    format(self.save_dir,args.data_name, args.model_name, args.epochs, args.frac,
                           args.iid, args.local_ep, args.local_bs))