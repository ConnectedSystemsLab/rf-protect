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
    start_x = np.random.uniform(low=-2.5, high=2.5)
    finish_x = np.random.uniform(low=-2.5, high=2.5)
    start_y = np.random.uniform(low=-2.5, high=2.5)
    finish_y = np.random.uniform(low=-2.5, high=2.5)
    traj_x = np.linspace(start_x, finish_x, 50).reshape(50, 1)
    traj_y = np.linspace(start_y, finish_y, 50).reshape(50, 1)
    traj = np.concatenate((traj_x, traj_y), 1)
    scipy.io.savemat('./check_tmp/ul_motion/{}.mat'.format(i),  {'M': traj})