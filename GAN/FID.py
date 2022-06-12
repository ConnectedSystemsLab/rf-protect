import matplotlib
matplotlib.use('Agg')
import os
from argparse import ArgumentParser
import scipy
import random

import numpy as np
import numpy.matlib
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision.utils import save_image
from torch.autograd import Variable
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm
from c_rnn_gan import Generator, Discriminator, classifier
import utils
from config import parsers
import matplotlib.pyplot as plt


# calculate frechet inception distance
def calculate_fid(act1, act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

# # real vs real
args = parsers()
# DATA_DIR_real = '../../DATA/_DATA_{}_{}/train'.format(args.time_length, args.trace_sample_interval)
# DATA_DIR_fake = './check_tmp/data'
DATA_DIR_real = '../../DATA/nsdi_DATA_c_50_10_50'
DATA_DIR_fake = '../../DATA/nsdi_DATA_c_50_10_50'
# Load Classifier Model
cla_net = classifier(num_feats=args.num_feats, use_cuda=False)
generator_stat = torch.load('./classifier_Model_1/Class_final.pt')['model_state_dict']
cla_net.load_state_dict(generator_stat)
# cla_net.cuda()

real_dataloader = utils.load_data(3000, DATA_DIR_real)
fake_dataloader = utils.load_data(3000, DATA_DIR_fake)
for i1, (batch_input1, labels1) in enumerate(real_dataloader):
    for i2, (batch_input2, labels2) in enumerate(fake_dataloader):
        real_batch_sz1 = len(batch_input1)
        real_batch_sz2 = len(batch_input2)
        states1 = cla_net.init_hidden(real_batch_sz1)
        states2 = cla_net.init_hidden(real_batch_sz2)
        with torch.no_grad():
            feature_real = cla_net.get_feature(batch_input1.float(), states1)
            # torch.cat([feature_fake, feature_real], dim=0)
            feature_fake = cla_net.get_feature(batch_input2.float(), states2)
            # feature_fake = cla_net.get_feature(torch.squeeze(batch_input2, dim=1), states2)
        break
    break
fid_real = calculate_fid(np.array(feature_real), np.array(feature_fake))
print("Real VS Real: ", 1)
#
# # real vs generate
args = parsers()
# DATA_DIR_real = '../../DATA/_DATA_{}_{}/train'.format(args.time_length, args.trace_sample_interval)
DATA_DIR_fake = './check_tmp/data_fake'
DATA_DIR_real = '../../DATA/nsdi_DATA_c_50_10_50'
# DATA_DIR_fake = '../../DATA/DATA2'
# Load Classifier Model
cla_net = classifier(num_feats=args.num_feats, use_cuda=False)
generator_stat = torch.load('./classifier_Model_1/Class_final.pt')['model_state_dict']
cla_net.load_state_dict(generator_stat)
# cla_net.cuda()

real_dataloader = utils.load_data(3000, DATA_DIR_real)
fake_dataloader = utils.load_data(3000, DATA_DIR_fake)
for i1, (batch_input1, labels1) in enumerate(real_dataloader):
    for i2, (batch_input2, labels2) in enumerate(fake_dataloader):
        real_batch_sz1 = len(batch_input1)
        real_batch_sz2 = len(batch_input2)
        states1 = cla_net.init_hidden(real_batch_sz1)
        states2 = cla_net.init_hidden(real_batch_sz2)
        with torch.no_grad():
            feature_real = cla_net.get_feature(batch_input1.float(), states1)
            # torch.cat([feature_fake, feature_real], dim=0)
            # feature_fake = cla_net.get_feature(batch_input2.float(), states2)
            feature_fake = cla_net.get_feature(torch.squeeze(batch_input2, dim=1), states2)
        break
    break
fid = calculate_fid(np.array(feature_real), np.array(feature_fake))
print("Real VS GAN: ", fid/fid_real)


# # real vs real_single_repeat
args = parsers()
# DATA_DIR_real = '../../DATA/_DATA_{}_{}/train'.format(args.time_length, args.trace_sample_interval)
# DATA_DIR_fake = './check_tmp/data'
DATA_DIR_real = '../../DATA/nsdi_DATA_c_50_10_50'
DATA_DIR_fake = '../../DATA/nsdi_DATA_c_50_10_50'
# Load Classifier Model
cla_net = classifier(num_feats=args.num_feats, use_cuda=False)
generator_stat = torch.load('./classifier_Model_1/Class_final.pt')['model_state_dict']
cla_net.load_state_dict(generator_stat)
# cla_net.cuda()

real_dataloader = utils.load_data(3000, DATA_DIR_real)
fake_dataloader = utils.load_data(3000, DATA_DIR_fake)
for i1, (batch_input1, labels1) in enumerate(real_dataloader):
    for i2, (batch_input2, labels2) in enumerate(fake_dataloader):
        real_batch_sz1 = len(batch_input1)
        real_batch_sz2 = len(batch_input2)
        states1 = cla_net.init_hidden(real_batch_sz1)
        states2 = cla_net.init_hidden(real_batch_sz2)
        with torch.no_grad():
            feature_real = cla_net.get_feature(batch_input1.float(), states1)
            # torch.cat([feature_fake, feature_real], dim=0)
            feature_fake = cla_net.get_feature(batch_input2[1].unsqueeze(0).repeat(3000,1,1).float(), states2)
            # feature_fake = cla_net.get_feature(torch.squeeze(batch_input2, dim=1), states2)
        break
    break
fid = calculate_fid(np.array(feature_real), np.array(feature_fake))
print("Real VS Real Single Repeat: ", fid/fid_real)
#


# real vs UL_motion
args = parsers()
# DATA_DIR_real = '../../DATA/_DATA_{}_{}/train'.format(args.time_length, args.trace_sample_interval)
DATA_DIR_fake = './check_tmp/ul_motion/'
DATA_DIR_real = '../../DATA/nsdi_DATA_c_50_10_50'
# DATA_DIR_fake = '../../DATA/DATA2'

real_dataloader = utils.load_data(3000, DATA_DIR_real)
fake_dataloader = utils.load_data(3000, DATA_DIR_fake)
for i1, (batch_input1, labels1) in enumerate(real_dataloader):
    for i2, (batch_input2, labels2) in enumerate(fake_dataloader):
        real_batch_sz1 = len(batch_input1)
        real_batch_sz2 = len(batch_input2)
        states1 = cla_net.init_hidden(real_batch_sz1)
        states2 = cla_net.init_hidden(real_batch_sz2)
        with torch.no_grad():
            feature_real = cla_net.get_feature(batch_input1.float(), states1)
            # torch.cat([feature_fake, feature_real], dim=0)
            # feature_fake = cla_net.get_feature(batch_input2.float(), states2)
            feature_fake = cla_net.get_feature(torch.squeeze(batch_input2, dim=1).float(), states2)
        break
    break
fid = calculate_fid(np.array(feature_real), np.array(feature_fake))
print("Real VS UL_motion: ", fid/fid_real)

# real vs random
args = parsers()
# DATA_DIR_real = '../../DATA/_DATA_{}_{}/train'.format(args.time_length, args.trace_sample_interval)
DATA_DIR_fake = './check_tmp/random_data'
DATA_DIR_real = '../../DATA/nsdi_DATA_c_50_10_50'
# DATA_DIR_fake = '../../DATA/DATA2'

real_dataloader = utils.load_data(3000, DATA_DIR_real)
fake_dataloader = utils.load_data(3000, DATA_DIR_fake)
for i1, (batch_input1, labels1) in enumerate(real_dataloader):
    for i2, (batch_input2, labels2) in enumerate(fake_dataloader):
        real_batch_sz1 = len(batch_input1)
        real_batch_sz2 = len(batch_input2)
        states1 = cla_net.init_hidden(real_batch_sz1)
        states2 = cla_net.init_hidden(real_batch_sz2)
        with torch.no_grad():
            feature_real = cla_net.get_feature(batch_input1.float(), states1)
            # torch.cat([feature_fake, feature_real], dim=0)
            # feature_fake = cla_net.get_feature(batch_input2.float(), states2)
            feature_fake = cla_net.get_feature(torch.squeeze(batch_input2, dim=1).float(), states2)
        break
    break
fid = calculate_fid(np.array(feature_real), np.array(feature_fake))
print("Real VS Random: ", fid/fid_real)


