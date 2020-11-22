import matplotlib
matplotlib.use('Agg')
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
DATA_DIR_TRN = '../../DATA/DATA_2000_10/train'
DATA_DIR_VAL = '../../DATA/DATA_2000_10/test'

G_FN = 'c_rnn_gan_g.pth'
D_FN = 'c_rnn_gan_d.pth'
gloss_array = []
dloss_array = []

MAX_GRAD_NORM = 5.0
BATCH_SIZE = 32
MAX_EPOCHS = 500
L2_DECAY = 1.0

MAX_SEQ_LEN = 300

PERFORM_LOSS_CHECKING = False
FREEZE_G = False
FREEZE_D = False

NUM_DUMMY_TRN = 256   # 训练数据集总共256
NUM_DUMMY_VAL = 128   # 验证数据集总共128

EPSILON = 1e-40 # value to use to approximate zero (to prevent undefined results)


def get_accuracy(logits_real, logits_gen):
    ''' Discriminator accuracy
    '''
    real_corrects = (logits_real > 0.5).sum()
    gen_corrects = (logits_gen < 0.5).sum()

    acc = (real_corrects + gen_corrects) / (len(logits_real) + len(logits_gen))
    return acc.item()


class GLoss(nn.Module):
    ''' C-RNN-GAN generator loss
    '''
    def __init__(self):
        super(GLoss, self).__init__()

    def forward(self, a, logits_gen):
        logits_gen = torch.clamp(logits_gen, EPSILON, 1.0)
        batch_loss = -torch.log(logits_gen)

        return torch.mean(batch_loss)


class DLoss(nn.Module):
    ''' C-RNN-GAN discriminator loss
    '''
    def __init__(self, label_smoothing=False):
        super(DLoss, self).__init__()
        self.label_smoothing = label_smoothing

    def forward(self, logits_real, logits_gen):
        ''' Discriminator loss

        logits_real: logits from D, when input is real
        logits_gen: logits from D, when input is from Generator

        loss = -(ylog(p) + (1-y)log(1-p))

        '''
        logits_real = torch.clamp(logits_real, EPSILON, 1.0)
        d_loss_real = -torch.log(logits_real)

        if self.label_smoothing:
            "Label Smoothing"
            p_fake = torch.clamp((1 - logits_real), EPSILON, 1.0)
            d_loss_fake = -torch.log(p_fake)
            d_loss_real = 0.9*d_loss_real + 0.1*d_loss_fake

        logits_gen = torch.clamp((1 - logits_gen), EPSILON, 1.0)
        d_loss_gen = -torch.log(logits_gen)

        batch_loss = d_loss_real + d_loss_gen



        return torch.mean(batch_loss)


def control_grad(model, freeze=True):
    ''' Freeze/unfreeze optimization of model
    '''
    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    else: # unfreeze
        for param in model.parameters():
            param.requires_grad = True


def check_loss(model, loss):
    ''' Check loss and control gradients if necessary
    '''
    control_grad(model['g'], freeze=False)
    control_grad(model['d'], freeze=False)

    if loss['d'] == 0.0 and loss['g'] == 0.0:
        print('Both G and D train loss are zero. Exiting.')
        return False
    elif loss['d'] == 0.0: # freeze D
        control_grad(model['d'], freeze=True)
    elif loss['g'] == 0.0: # freeze G
        control_grad(model['g'], freeze=True)
    # elif loss['g'] < 2.0 or loss['d'] < 2.0:
    #     control_grad(model['d'], freeze=True)
        if loss['g']*0.7 > loss['d']:
            control_grad(model['g'], freeze=True)

    return True


def run_training(model, optimizer, criterion, dataloader, ep, freeze_g=False, freeze_d=False):
    args = parsers()
    ''' Run single training epoch
    '''

    loss = {
        'g': 10.0,
        'd': 10.0
    }

    num_feats = model['g'].num_feats
    cuda = True if (torch.cuda.is_available() and args.GPU) else False
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    model['g'].train()
    model['d'].train()

    adversarial_loss = torch.nn.BCELoss()

    g_loss_total = 0.0
    d_loss_total = 0.0
    num_corrects = 0
    num_sample = 0

    log_sum_real = 0.0
    log_sum_gen = 0.0

    for i, (batch_input, labels) in enumerate(dataloader):

        real_batch_sz = len(batch_input)
        batch_input = batch_input.type(torch.FloatTensor)  # from 64->32

        # get initial states
        # each batch is independent i.e. not a continuation of previous batch
        # so we reset states for each batch
        # POSSIBLE IMPROVEMENT: next batch is continuation of previous batch
        g_states = model['g'].init_hidden(real_batch_sz)
        d_state = model['d'].init_hidden(real_batch_sz)
        labels = Variable(labels.type(LongTensor))
        # ---------------------
        #  Train Discriminator
        # ---------------------
        if not freeze_d:
            optimizer['d'].zero_grad()
        # feed real and generated input to discriminator
        z = torch.empty([real_batch_sz, args.latent_dim, args.num_feats]).normal_(0, 1)# random vector
        gen_labels = Variable(LongTensor(np.random.randint(0, 5, real_batch_sz)))

        d_logits_real, _, _ = model['d'](batch_input, d_state, labels)
        g_feats, _ = model['g'](z, g_states, gen_labels)
        # need to detach from operation history to prevent backpropagating to generator
        d_logits_gen, _, _ = model['d'](g_feats.detach(), d_state, gen_labels)

        # ####### calculate loss, backprop, and update weights of D####### #
        # valid = Variable(FloatTensor(real_batch_sz, 1).fill_(1.0), requires_grad=False).view(-1)
        # fake = Variable(FloatTensor(real_batch_sz, 1).fill_(0.0), requires_grad=False).view(-1)
        # d_real_loss = adversarial_loss(d_logits_real, valid)
        # d_fake_loss = adversarial_loss(d_logits_gen, fake)
        # loss['d'] = d_real_loss + d_fake_loss

        loss['d'] = criterion['d'](d_logits_real, d_logits_gen)

        log_sum_real += d_logits_real.sum().item() # 将所有batach的得分都加起来
        log_sum_gen += d_logits_gen.sum().item()

        if not freeze_d:
            loss['d'].backward()
            nn.utils.clip_grad_norm_(model['d'].parameters(), max_norm=MAX_GRAD_NORM)
            optimizer['d'].step()

        # -----------------
        #  Train Generator
        # -----------------

        if not freeze_g:
            optimizer['g'].zero_grad()
        # prepare inputs
        z = torch.empty([real_batch_sz, args.latent_dim, args.num_feats]).normal_(0, 1)# random vector
        gen_labels = Variable(LongTensor(np.random.randint(0, 5, real_batch_sz)))
        # feed inputs to generator
        g_feats, _ = model['g'](z, g_states, gen_labels)  # 32*8*1
        # feed real and generated input to discriminator
        d1, d_feats_real, _ = model['d'](batch_input, d_state, labels) # 32*8*1,
        d2, d_feats_gen, _ = model['d'](g_feats, d_state, gen_labels) # 这里全部都是取的lstm的output

        # calculate loss, backprop, and update weights of G
        if args.feature_matching:
            loss['g'] = criterion['g'](d_feats_real, d_feats_gen)
        else:
            valid = Variable(FloatTensor(real_batch_sz, 1).fill_(1.0), requires_grad=False).view(-1)
            loss['g'] = adversarial_loss(d2, valid)
        if not freeze_g:
            loss['g'].backward()
            # nn.utils.clip_grad_norm_(model['g'].parameters(), max_norm=MAX_GRAD_NORM)
            optimizer['g'].step()

        # print(
        #     "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
        #     % (ep, args.num_epochs, i, len(dataloader), loss['d'].item(), loss['g'].item())
        # )
        # ---------------------
        #  Plot generated traces
        # ---------------------

        batches_done = ep * len(dataloader) + i
        if batches_done % args.sample_interval == 0:
            # print(
            #     "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            #     % (ep, args.num_epochs, i, len(dataloader), loss['d'].item(), loss['g'].item())
            # )
            gen_imgs = g_feats.cpu()
            gen_imgs = gen_imgs.detach().numpy()
            # generate_sample(model['g'], batches_done)
            # fig = plt.figure(figsize=(17, 17))
            # for i_s in range(1, 26):
            #     plt.subplot(5, 5, i_s)
            #     plt.scatter(gen_imgs[i_s-1, :args.seq_len, 0], gen_imgs[i_s-1, :args.seq_len, 1], 2)
            # fig.suptitle('Epoch:{}'.format(ep), fontsize=30)

            # plt.scatter(gen_imgs[1, :args.seq_len, 0], gen_imgs[1, :args.seq_len, 1], 2)
            # plt.title('epoch:{}'.format(ep))
            # np.save(args.data_path+'/{}.npy'.format(batches_done), gen_imgs)
            # plt.savefig(args.pics_path+'/{}_scatter.jpg'.format(batches_done))
            # plt.close()

            # fig = plt.figure(figsize=(17, 17))
            # for i_s in range(1, 26):
            #     plt.subplot(5, 5, i_s)
            #     plt.plot(gen_imgs[i_s-1, :args.seq_len, 0], gen_imgs[i_s-1, :args.seq_len, 1], 2)
            # fig.suptitle('epoch:{}'.format(ep), fontsize=30)
            # plt.plot(gen_imgs[1, :args.seq_len, 0], gen_imgs[1, :args.seq_len, 1], 2)
            # plt.title('epoch:{}'.format(ep))
            # plt.xlim(-1.01, 1.01)
            # plt.ylim(-1.01, 1.01)
            # plt.savefig(args.pics_path+'/{}_plot.jpg'.format(batches_done))
            # plt.close()

            # save_image(gen_imgs[:16], args.pics_path+'/{}.png'.format(batches_done), nrow=4, normalize=True)
            gloss_array.append(loss['g'].item())
            dloss_array.append(loss['d'].item())
            dloss_plot_path = args.stats_path+'/dloss.png'
            plt.plot(dloss_array, color='red', linewidth=2.0)
            plt.ylabel("d-loss")
            plt.xlabel("epoch")
            plt.savefig(dloss_plot_path)
            plt.close()
            np.save(args.stats_path+'/d_loss.npy', dloss_array)

            gloss_plot_path = args.stats_path+'/gloss.png'
            plt.plot(gloss_array, color='red', linewidth=2.0)
            plt.ylabel("g-loss")
            plt.xlabel("epoch")
            plt.savefig(gloss_plot_path)
            plt.close()
            np.save(args.stats_path+'/g_loss.npy', gloss_array)

        g_loss_total += loss['g'].item()
        d_loss_total += loss['d'].item()
        num_corrects += (d_logits_real > 0.5).sum().item() + (d_logits_gen < 0.5).sum().item()
        num_sample += real_batch_sz

    g_loss_avg, d_loss_avg = 0.0, 0.0
    d_acc = 0.0
    if num_sample > 0:
        g_loss_avg = g_loss_total / num_sample
        d_loss_avg = d_loss_total / num_sample
        d_acc = 100 * num_corrects / (2 * num_sample) # 2 because (real + generated)

        print("Trn: ", log_sum_real / num_sample, log_sum_gen / num_sample)

    if ep % 10 == 0:

        path1 = args.Model_path+'/g_epoch_{}.pt'.format(str(ep))
        torch.save({
            'epoch': ep,
            'model_state_dict': model['g'].state_dict(),
        }, path1)

        path1 = args.Model_path+'/d_epoch_{}.pt'.format(str(ep))
        torch.save({
            'epoch': ep,
            'model_state_dict': model['d'].state_dict(),
        }, path1)

    return model, g_loss_avg, d_loss_avg, d_acc


def generate_sample(g_model, batches_done):
    ''' Sample from generator
    '''
    n_row = 5
    num_sample = n_row*n_row
    args = parsers()
    z = torch.empty([num_sample, args.latent_dim, args.num_feats]).normal_(0, 1) # random vector

    g_states = g_model.init_hidden(num_sample)
    LongTensor = torch.cuda.LongTensor
    gen_labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    gen_labels = Variable(LongTensor(gen_labels))

    g_feats, _ = g_model(z, g_states, gen_labels)
    gen_imgs = g_feats.cpu()
    gen_imgs = gen_imgs.detach().numpy()
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
    np.save(args.data_path+'/{}.npy'.format(batches_done), gen_imgs)
    plt.savefig(args.pics_path+'/{}_plot.jpg'.format(batches_done))
    plt.close()
    # save_image(g_feats.data, args.pics_path+ "%d.png" % batches_done, nrow=n_row, normalize=True)
    return


def main(args):
    ''' Training sequence
    '''
    trn_dataloader = utils.load_data(args.batch_size, DATA_DIR_TRN)
    val_dataloader = utils.load_data(args.batch_size, DATA_DIR_TRN)

    # First checking if GPU is available
    train_on_gpu = torch.cuda.is_available() and args.GPU
    if train_on_gpu:
        print('Training on GPU.')
    else:
        print('No GPU available, training on CPU.')

    model = {
        'g': Generator(num_feats=args.num_feats, use_cuda=train_on_gpu),
        'd': Discriminator(num_feats=args.num_feats, use_cuda=train_on_gpu)
    }

    optimizer = {
        # 'g': optim.SGD(model['g'].parameters(), G_LRN_RATE, weight_decay=L2_DECAY),
        'g': optim.Adam(model['g'].parameters(), args.g_lr),
        'd': optim.Adam(model['d'].parameters(), args.d_lr)
    }

    criterion = {
        'g': nn.MSELoss(reduction='sum'),
        'd': DLoss(label_smoothing=args.label_smoothing)
    }

    if train_on_gpu:
        model['g'].cuda()
        model['d'].cuda()

    # ---------------------
    #  Pre training
    # ---------------------

    if not args.no_pretraining:
        for ep in range(args.pretraining_epochs):
            model, trn_g_loss, trn_d_loss, trn_acc = \
                run_training(model, optimizer, criterion, trn_dataloader, ep, freeze_g=True)
            # val_g_loss, val_d_loss, val_acc = run_validation(model, criterion, val_dataloader)

            print("Pretraining Epoch %d/%d\n"
                  "\t[Training] G_loss: %0.8f, D_loss: %0.8f, D_acc: %0.2f\n"
                  "############################################################" %
                  (ep+1, args.num_epochs, trn_g_loss, trn_d_loss, trn_acc))

        for ep in range(args.pretraining_epochs):
            model, trn_g_loss, trn_d_loss, trn_acc = \
                run_training(model, optimizer, criterion, trn_dataloader, ep, freeze_d=True)
            # val_g_loss, val_d_loss, val_acc = run_validation(model, criterion, val_dataloader)

            print("Pretraining Epoch %d/%d\n"
                  "\t[Training] G_loss: %0.8f, D_loss: %0.8f, D_acc: %0.2f\n"
                  "############################################################" %
                  (ep+1, args.num_epochs, trn_g_loss, trn_d_loss, trn_acc))

    # ---------------------
    #  Training
    # ---------------------
    flag = False
    for ep in range(args.num_epochs):
        model, trn_g_loss, trn_d_loss, trn_acc = run_training(model, optimizer, criterion, trn_dataloader, ep, freeze_d=flag)
        if args.freezing:
            if trn_acc > 95:
                flag = True
                print("Freeze D!")
            else:
                flag = False
        generate_sample(model['g'], ep)
        # val_g_loss, val_d_loss, val_acc = run_validation(model, criterion, val_dataloader)

        # print("Epoch %d/%d\n"
        #       "\t[Training] G_loss: %0.8f, D_loss: %0.8f, D_acc: %0.2f\n"
        #       "\t[Validation] G_loss: %0.8f, D_loss: %0.8f, D_acc: %0.2f\n"
        #       "############################################################" %
        #       (ep+1, args.num_epochs, trn_g_loss, trn_d_loss, trn_acc,
        #        val_g_loss, val_d_loss, val_acc))
        print("Epoch %d/%d\n"
              "\t[Training] G_loss: %0.8f, D_loss: %0.8f, D_acc: %0.2f\n"
              "############################################################" %
              (ep+1, args.num_epochs, trn_g_loss, trn_d_loss, trn_acc))

        # sampling (to check if generator really learns)


if __name__ == "__main__":

    ARGS = parsers()
    MAX_SEQ_LEN = ARGS.seq_len
    BATCH_SIZE = ARGS.batch_size
    FREEZE_G = ARGS.freeze_g
    FREEZE_D = ARGS.freeze_d

    utils.mkr(ARGS.Model_path)
    utils.mkr(ARGS.data_path)
    utils.mkr(ARGS.pics_path)
    utils.mkr(ARGS.stats_path)
    DATA_DIR_TRN = '../../DATA/_DATA_{}_{}/train'.format(ARGS.time_length, ARGS.trace_sample_interval)
    DATA_DIR_VAL = '../../DATA/_DATA_{}_{}/test'.format(ARGS.time_length, ARGS.trace_sample_interval)
    print(ARGS)
    main(ARGS)
