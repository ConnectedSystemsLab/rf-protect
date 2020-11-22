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

from c_rnn_gan import Generator, Discriminator
import utils
from config import parsers
import matplotlib.pyplot as plt
from torch.autograd import Variable

args = parsers()
train_on_gpu = torch.cuda.is_available() and args.GPU

generator = Generator(num_feats=args.num_feats, use_cuda=True)

generator_stat = torch.load('./Trained Model/g_epoch_1020.pt')['model_state_dict']
generator.load_state_dict(generator_stat)
generator.cuda()
# generator.eval()
LongTensor = torch.cuda.LongTensor
FloatTensor = torch.cuda.FloatTensor

for i in range(64):
    g_states = generator.init_hidden(1)
    z = torch.empty([1, args.latent_dim, args.num_feats]).uniform_()
    gen_labels = Variable(LongTensor(np.random.randint(4, 5, 1))).cuda()
    g_feats, _ = generator(z, g_states, gen_labels)

    gen_imgs = g_feats.cpu()
    gen_imgs = gen_imgs.detach().numpy()

    # fig = plt.figure(figsize=(17, 17))
    # for i_s in range(1, 10):
    #     plt.subplot(3, 3, i_s)
    #     plt.scatter(gen_imgs[i_s-1, :args.seq_len, 0], gen_imgs[i_s-1, :args.seq_len, 1], 2)
    # fig.suptitle('Epoch:{}'.format(ep), fontsize=30)
    plt.scatter(gen_imgs[0, :args.seq_len, 0], gen_imgs[0, :args.seq_len, 1], 2)
    plt.title('epoch:{}'.format(i))
    # np.save(args.data_path+'/{}.npy'.format(0), gen_imgs)
    plt.savefig('./check_tmp/{}_scatter.jpg'.format(i))
    plt.close()
    plt.show()

    plt.plot(gen_imgs[0, :args.seq_len, 0], gen_imgs[0, :args.seq_len, 1], 2)
    plt.title('epoch:{}'.format(0))
    # plt.xlim(-1.01, 1.01)
    # plt.ylim(-1.01, 1.01)
    plt.savefig('./check_tmp/{}_plot.jpg'.format(i))
    plt.close()
