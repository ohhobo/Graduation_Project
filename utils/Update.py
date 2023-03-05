import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import math
import logging

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """
    #Dataset可以是任何东西，但它始终包含一个__len__函数(通过Python中的标准函数len调用）和一个用来索引到内容中的__getitem__函数。
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)#数据列表长度即数据集的样本数量

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)#获取image和label的数量

class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, i):
        self.args = args
        #获得数据加载器
        self.trainloader, self.validloader = self.train_val_test(dataset, list(idxs))
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.i = i

    def train_val_test(self, dataset, idxs):
        '''
        输入数据集和索引，按照6:2:2来划分。
        注意到在指定batchsize的时候，除了训练集是从args参数里指定的，val和test都是取了总数的十分之一。
        '''
        idxs_train = idxs[:int(0.75*len(idxs))]
        idxs_val = idxs[int(0.75*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),batch_size=math.ceil(len(idxs_val)/10),shuffle=False)

        return trainloader, validloader

    #本地权重更新，输入模型和全局更新的回合数，输出更新后的权重和损失平均值
    def update_weights(self, model, global_round):
        model.train()
        epoch_loss = []

        if self.args.opt == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(),lr=self.args.lr,momentum=0.5)
        elif self.args.opt == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):

                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs,labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    logging.info('| Global Round : {} | Client : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, self.i, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                                            100. * batch_idx / len(self.trainloader), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        #每经过一次本地轮次，统计当前的loss，用于最后的平均损失统计
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    #评估函数 验证集
    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.validloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss

def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = nn.CrossEntropyLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=8,
                            shuffle=False)

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
    return accuracy, loss