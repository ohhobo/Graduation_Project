import logging
import os
import time
import warnings
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from datetime import datetime
import models
import numpy as np
from models.CNN_2d import CNN
from utils.Update import LocalUpdate,test_inference
from tqdm import tqdm
import copy
from train import parse_args
from utils.train_federated import get_dataset

if __name__ == '__main__':
    args = parse_args()
    train_dataset, test_dataset, user_groups = get_dataset(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.model_name == 'cnn_2d':
        if args.data_name == 'CWRUSTFT':
            model = CNN()
        elif args.data_name == 'PUSTFT':
            model =CNN(out_channel=3)
    else:
        exit('Error: unrecognized model')
    model.to(device)
    if args.iid:
        sub_dir = args.train_type+'_'+args.model_name+'_'+args.data_name + '_' + 'iid' + '_' + datetime.strftime(datetime.now(), '%m%d')
    else:
        sub_dir = args.train_type+'_'+args.model_name+'_'+args.data_name + '_' + 'non_iid' + '_' + datetime.strftime(datetime.now(), '%m%d')
    save_dir = os.path.join(args.checkpoint_dir, sub_dir)
    model_path = os.path.join(save_dir,'fed_CWRUSTFT_cnn_2d_1_C[0.5]_iid[1]_E[10]_B[8]_final_model.pth')
    model_state_dict = torch.load(model_path)
    model.load_state_dict(model_state_dict)

    loss, total, correct = 0.0, 0.0, 0.0
    criterion = nn.CrossEntropyLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=8,
                            shuffle=False)
    model.eval()
    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Test Accuracy: {:.2f}%".format(100*accuracy))
