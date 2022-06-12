import matplotlib
matplotlib.use('Agg')
import os
from argparse import ArgumentParser
import scipy
import scipy.io
import random
import torch.multiprocessing as mp
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
from scipy.ndimage import gaussian_filter1d


data_path = './check_tmp/data_fake'
pics_path = './Pics'

num_trace = len([name for name in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, name))])
for idx in range(num_trace):
    plt.figure()
    data = scipy.io.loadmat(data_path+'/{}.mat'.format(idx))['M']
    # Plot and save
    plt.plot(gaussian_filter1d(data[:, 0], 3), gaussian_filter1d(data[:, 1], 3), linewidth=5)
    # plt.show()
    plt.xlim(-1.8,1.8)
    plt.ylim(-1.8,1.8)
    plt.savefig(pics_path+'/{}.png'.format(idx))
    plt.close()

