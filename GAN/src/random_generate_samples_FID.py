import matplotlib
matplotlib.use('Agg')
import os
from argparse import ArgumentParser
import scipy
import random
import scipy.io
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision.utils import save_image
from torch.autograd import Variable

from c_rnn_gan import Generator, Discriminator
import utils
from config import parsers
import matplotlib.pyplot as plt


num_data = 3000
for i in range(num_data):
    x = np.random.uniform(low=-2.5, high=2.5, size=(50,1))
    y = np.random.uniform(low=-2.5, high=2.5, size=(50,1))
    traj = np.concatenate((x, y), 1)
    scipy.io.savemat('./check_tmp/random_data/{}.mat'.format(i),  {'M': traj})