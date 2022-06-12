import os
from argparse import ArgumentParser
import scipy
import random

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

args = parsers()
DATA_DIR_TRN = '../../DATA/_DATA_{}_{}/train'.format(args.time_length, args.trace_sample_interval)
dataloader = utils.load_data(args.batch_size, DATA_DIR_TRN)

for i, (batch_input, labels) in enumerate(dataloader):
    batch_input = batch_input[0]
    value = max((batch_input[:, 0].max()-batch_input[:, 0].min()), (batch_input[:, 1].max()-batch_input[:, 1].min()))
    plt.plot()