"""
Plot the .npy file downloaded from the server
"""
import matplotlib
matplotlib.use('Agg')
import os
from argparse import ArgumentParser
import scipy
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

data_path = '../../Download/data/trace-by-gan1/data/'
pics_path = '../../Download/pics/con1/'

def generate_sample(gen_imgs, batches_done):
    n_row = 5
    num_sample = n_row*n_row
    args = parsers()

    fig = plt.figure(figsize=(17, 17))

    for i_s in range(1, 26):
        ax = plt.subplot(5, 5, i_s)
        plt.plot(gen_imgs[i_s-1, :args.seq_len, 0], gen_imgs[i_s-1, :args.seq_len, 1], 2)
        value = max(gen_imgs[i_s-1, :args.seq_len, 0].max()-gen_imgs[i_s-1, :args.seq_len, 0].min(), gen_imgs[i_s-1, :args.seq_len, 1].max()-gen_imgs[i_s-1, :args.seq_len, 1].min())

        if value < 0.8685:
            r = 0
        elif value < 1.334:
            r = 1
        elif value < 1.896:
            r = 2
        elif value < 3.132:
            r = 3
        else:
            r = 4
        ax.set_title('{}-{:.2f}'.format(r, value))
    fig.suptitle('epoch:{}'.format(batches_done), fontsize=30)
    plt.savefig(pics_path+'/{}_plot.jpg'.format(batches_done))
    plt.close()
    # save_image(g_feats.data, args.pics_path+ "%d.png" % batches_done, nrow=n_row, normalize=True)
    return


def plot_sample(index1, index2):
    for i in range(index1, index2):
        imgs = np.load(data_path+'{}.npy'.format(i))
        generate_sample(imgs, i)
    print('finish {} to {}'.format(index1, index2))


if __name__ == "__main__":
    thread_num = 4

    num_trace = len([name for name in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, name))])
    # for i in range(num_trace-1, 0, -1):
    #     imgs = np.load(data_path+'{}.npy'.format(i))
    #     generate_sample(imgs, i)
    processes = []
    interval = int(num_trace/thread_num)
    for rank in range(thread_num):
        index1 = rank*interval
        if rank == thread_num-1:
            index2 = num_trace
        else:
            index2 = (rank+1)*interval
        p = mp.Process(target=plot_sample, args=(index1, index2))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()