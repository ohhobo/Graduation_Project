import os
import pandas as pd
import numpy as np
from scipy.io import loadmat
from datasets.MatrixDatasets import dataset
from datasets.matrix_aug import *
from tqdm import tqdm
import pickle
from scipy import signal
from sklearn.model_selection import train_test_split

signal_size = 1024

Hdata = ['K001','K002','K003','K004','K005']#健康 label:0
IRdata = ['KI14','KI17','KI21','KI16','KI18']#内圈 label:1
ORdata = ['KA04','KA15','KA16','KA22','KA30']#外圈 label:2

#working condition
WC = ["N15_M07_F10","N09_M07_F10","N15_M01_F10","N15_M07_F04"]
state = WC[0] #WC[0] can be changed to different working states

def  STFT(fl):
    f, t, Zxx = signal.stft(fl, nperseg=64)
    img = np.abs(Zxx) / len(Zxx)
    return img

#generate Training Dataset and Testing Dataset
def get_files(root, test=False):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''
    data = []
    lab = []

    for k in tqdm(range(len(Hdata))):
        name1 = state+"_"+Hdata[k]+"_1"
        path1=os.path.join(root,Hdata[k],name1+".mat")
        data1, lab1= data_load(path1,name=name1,label=0)
        data +=data1
        lab +=lab1

    for k in tqdm(range(len(IRdata))):
        name2 = state+"_"+IRdata[k]+"_1"
        path2=os.path.join(root,IRdata[k],name2+".mat")
        data2, lab2= data_load(path2,name=name2,label=1)
        data +=data2
        lab +=lab2

    for k in tqdm(range(len(ORdata))):
        name3 = state+"_"+ORdata[k]+"_1"
        path3=os.path.join(root,ORdata[k],name3+".mat")
        data3, lab3= data_load(path3,name=name3,label=2)
        data +=data3
        lab +=lab3

    return [data,lab]

def data_load(filename,name,label):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    '''
    fl = loadmat(filename)[name]
    fl = fl[0][0][2][0][6][2]  #Take out the data
    fl = fl.reshape(-1,)
    data=[] 
    lab=[]
    start,end=0,signal_size
    while end<=fl.shape[0]:
        x = fl[start:end]
        imgs = STFT(x)
        data.append(imgs)
        lab.append(label)
        start +=signal_size
        end +=signal_size

    return data, lab

def data_transforms(dataset_type="train", normlize_type="-1-1"):
    transforms = {
        'train': Compose([
        ReSize(size=10.0),
        Reshape(),
        Normalize(normlize_type),
        RandomScale(),
        RandomCrop(),
        Retype(),
    ]),
        'val': Compose([
        ReSize(size=10.0),
        Reshape(),
        Normalize(normlize_type),
        Retype(),
    ])
    }
    return transforms[dataset_type]
#--------------------------------------------------------------------------------------------------------------------
class PUSTFT(object):
    num_classes = 13
    inputchannel = 1

    def __init__(self, args, data_dir, normlizetype):
        self.args = args
        self.data_dir = data_dir
        self.normlizetype = normlizetype

    def data_preprare(self, test=False):
        if len(os.path.basename(self.data_dir).split('.')) == 2:
            with open(self.data_dir, 'rb') as fo:
                list_data = pickle.load(fo, encoding='bytes')
        else:
            list_data = get_files(self.data_dir, test)
            with open(os.path.join(self.data_dir, "PUSTFT.pkl"), 'wb') as fo:
                pickle.dump(list_data, fo)
        if self.args.train_type == 'train_utils':
            if test:
                test_dataset = dataset(list_data=list_data, test=True, transform=None)
                return test_dataset
            else:
                data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
                train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
                train_dataset = dataset(list_data=train_pd, transform=data_transforms('train',self.normlizetype))
                val_dataset = dataset(list_data=val_pd, transform=data_transforms('val',self.normlizetype))
                return train_dataset, val_dataset
        elif self.args.train_type == 'train_federated':#联邦学习
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, test_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            train_dataset = dataset(list_data=train_pd,transform=data_transforms('train',self.normlizetype))
            test_dataset = dataset(list_data=test_pd,transform=data_transforms('val',self.normlizetype))
            return train_dataset,test_dataset



