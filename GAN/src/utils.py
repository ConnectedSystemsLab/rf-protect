import os
from argparse import ArgumentParser
import scipy.io
import random
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import shutil


def mkr(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
    else:
        os.makedirs(path, exist_ok=True)

class TraceDataset(torch.utils.data.Dataset):

    def __init__(self, root, transform=None, train=True): #第一步初始化各个变量
        self.root = root
        self.train = train
        self.length = len([name for name in os.listdir(self.root) if os.path.isfile(os.path.join(self.root, name))])-1
    def __getitem__(self, idx):
        trace = scipy.io.loadmat(self.root+'/{}.mat'.format(idx+1))
        trace = np.array(trace['M'])
        self.trace = trace
        # trace = trace[:8]
        # while min([trace[:, 0].max()-trace[:, 0].min(), trace[:, 1].max()-trace[:, 1].min()]) < 0.001:
        #     print("retake!")
        #     idx_random = random.randint(1, self.__len__())
        #     trace = scipy.io.loadmat(self.root+'/{}.mat'.format(idx_random))
        #     trace = np.array(trace['data_save'])
        #     trace = trace[:8]
        # trace[:, 0] = (trace[:, 0]-trace[:, 0].min())/(trace[:, 0].max()-trace[:, 0].min())
        # trace[:, 1] = (trace[:, 1]-trace[:, 1].min())/(trace[:, 1].max()-trace[:, 1].min())
        # trace = 2*trace-1

        label = self.class_define()
        # self.trace[:, 0] = (200 - self.trace[:, 0])*0.15
        # self.trace = [self.trace[:, 0]*np.cos(np.radians(self.trace[:, 1])), self.trace[:, 0]*np.sin(np.radians(self.trace[:, 1]))]
        # self.trace = np.transpose(self.trace)
        self.trace[:, 0] = self.trace[:, 0] - np.mean(self.trace[:, 0])
        self.trace[:, 1] = self.trace[:, 1] - np.mean(self.trace[:, 1])
        return self.trace, label

    def __len__(self):
        return int(self.length)
        # return 10000

    def class_define(self):
        self.trace1 = self.trace
        value = max((self.trace1[:, 0].max()-self.trace1[:, 0].min()), (self.trace1[:, 1].max()-self.trace1[:, 1].min()))
        # if value < 2.109:
        #     return np.array([0])
        # elif value < 2.615:
        #     return np.array([1])
        # elif value < 2.963:
        #     return np.array([2])
        # elif value < 3.247:
        #     return np.array([3])
        # else:
        #     return np.array([4])
        # for 50_10_50
        if value < 2.1224:
            return np.array([0])
        elif value < 2.6281:
            return np.array([1])
        elif value < 2.9789:
            return np.array([2])
        elif value < 3.2586:
            return np.array([3])
        else:
            return np.array([4])

def load_data(batch_size=64, root="../images"):
    train_set = TraceDataset(root=root)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    return train_loader
