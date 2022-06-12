import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
import utils
import torch.nn.functional as F
from config_classifier import parsers
from c_rnn_gan import classifier


def run_training(net, optimizer, dataloader, ep):

    net.train()

    loss_array_ave = []
    correct = 0
    for i, (batch_input, labels) in enumerate(dataloader):
        labels = labels.view(-1).cuda().long()
        real_batch_sz = len(batch_input)
        states = net.init_hidden(real_batch_sz)
        y_pred = net(batch_input, states)
        loss = F.nll_loss(y_pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_array_ave.append(loss.item())
        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(labels.view_as(pred)).sum().item()
    loss_val = np.mean(loss_array_ave)
    print("Training epoch:{}, loss:{},  Accuracy: {}/{} ({:.0f}%)".format(ep, loss_val, correct, len(dataloader.dataset), 100. * correct / len(dataloader.dataset)))
    if ep % 200 == 0:
        path1 = args.Model_path+'/{}.pt'.format(str(ep))
        torch.save({
            'epoch': ep,
            'model_state_dict': net.state_dict(),
        }, path1)
    return net, loss_val


def run_testing(net, dataloader):

    net.eval()

    loss_array_ave = []
    correct = 0
    with torch.no_grad():
        for i, (batch_input, labels) in enumerate(dataloader):
            labels = labels.view(-1).cuda().long()
            real_batch_sz = len(batch_input)
            states = net.init_hidden(real_batch_sz)
            y_pred = net(batch_input, states)
            loss = F.nll_loss(y_pred, labels)
            loss_array_ave.append(loss.item())
            pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(labels.view_as(pred)).sum().item()
    loss_val = np.mean(loss_array_ave)
    print("Test loss:{}, Accuracy: {}/{} ({:.0f}%)\n".format(loss_val, correct, len(dataloader.dataset), 100. * correct / len(dataloader.dataset)))

    return loss_val


def main(args):
    trn_dataloader = utils.load_data(args.batch_size, '../../DATA/DATA_FID_50/train')
    tst_dataloader = utils.load_data(args.batch_size, '../../DATA/DATA_FID_50/test')
    train_on_gpu = torch.cuda.is_available() and args.GPU
    net = classifier(num_feats=args.num_feats, use_cuda=train_on_gpu)

    # generator_stat = torch.load('./Class_final.pt')['model_state_dict']
    # net.load_state_dict(generator_stat)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    loss_array_trn = []
    loss_array_tst = []
    lowest_test_val = 100000
    if train_on_gpu:
        net.cuda()
    net.double()
    for ep in range(args.epoch):
        net, loss_val_trn = run_training(net, optimizer, trn_dataloader, ep)
        loss_array_trn.append(loss_val_trn)

        loss_val_tst = run_testing(net, tst_dataloader)
        loss_array_tst.append(loss_val_tst)

        plt.plot(loss_array_trn, color='red', linewidth=2.0, label='Train loss')
        plt.plot(loss_array_tst, color='green', linewidth=2.0, label='Test loss')
        plt.ylabel("Loss")
        plt.xlabel("epoch")
        plt.legend()
        plt.title('Loss Analysis')
        plt.savefig(args.stats_path+'/loss.png')
        plt.close()
        np.save(args.stats_path+'/loss_trn.npy', loss_array_trn)
        np.save(args.stats_path+'/loss_tst.npy', loss_array_tst)

        if loss_val_tst < lowest_test_val:
            lowest_test_val = loss_val_tst
            path1 = args.Model_path+'/Class_final.pt'.format(str(ep))
            torch.save({
                'epoch': ep,
                'model_state_dict': net.state_dict(),
            }, path1)
if __name__ == "__main__":
    args = parsers()
    utils.mkr(args.Model_path)
    utils.mkr(args.stats_path)
    main(args)