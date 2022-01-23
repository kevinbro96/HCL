import argparse
import os
import pandas
import sys
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from model import Model
sys.path.append('..')
from vae import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Home device: {}'.format(device))


def my_mkdir(dir):
    try:
        os.mkdir(dir)
    except:
        pass


def gen_adv(net, vae, pos_1, out_1, criterion):
    pos_1 = pos_1.detach()
    with torch.no_grad():
        z, gx, _, _ = vae(pos_1)
    variable_bottle = Variable(z.detach(), requires_grad=True)
    adv_gx = vae(variable_bottle, True)
    pos_adv = adv_gx + (pos_1 - gx).detach()
    _, out_adv = net(pos_adv, adv=True)

    tmp_loss = criterion(out_1.detach(), out_adv)
    tmp_loss.backward()

    with torch.no_grad():
        sign_grad = variable_bottle.grad.data.sign()
        variable_bottle.data = variable_bottle.data + 0.1 * sign_grad
        adv_gx = vae(variable_bottle, True)
        pos_adv = adv_gx + (pos_1 - gx).detach()
    pos_adv.requires_grad = False
    pos_adv.detach()
    return pos_adv, gx


def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask


class HardCL(nn.Module):

    def __init__(self, tau_plus, batch_size, beta, estimator, temperature):
        super(HardCL, self).__init__()
        self.tau_plus = tau_plus
        self.batch_size = batch_size
        self.beta = beta
        self.estimator = estimator
        self.temperature = temperature

    def forward(self, out_1,out_2):

        out = torch.cat([out_1, out_2], dim=0)
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
        old_neg = neg.clone()
        mask = get_negative_mask(self.batch_size).to(device)
        neg = neg.masked_select(mask).view(2 * self.batch_size, -1)

        # pos score
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)

        # negative samples similarity scoring
        if self.estimator == 'hard':
            N = self.batch_size * 2 - 2
            imp = (self.beta * neg.log()).exp()
            reweight_neg = (imp * neg).sum(dim=-1) / imp.mean(dim=-1)
            Ng = (-self.tau_plus * N * pos + reweight_neg) / (1 - self.tau_plus)
            # constrain (optional)
            Ng = torch.clamp(Ng, min=N * np.e ** (-1 / self.temperature))
        elif self.estimator == 'easy':
            Ng = neg.sum(dim=-1)
        else:
            raise Exception('Invalid estimator selected. Please use any of [hard, easy]')

        # contrastive loss
        loss = (- torch.log(pos / (pos + Ng))).mean()

        return loss


def train(net, vae, data_loader, train_optimizer, temperature, estimator, tau_plus, beta):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    criterion = HardCL(tau_plus, batch_size, beta, estimator, temperature)
    for pos_1, pos_2, target in train_bar:
        pos_1, pos_2 = pos_1.to(device,non_blocking=True), pos_2.to(device,non_blocking=True)
        feature_1, out_1 = net(pos_1)
        pos_adv, gx = gen_adv(net, vae, pos_1, out_1.detach(), criterion)
        train_optimizer.zero_grad()
        feature_adv, out_adv = net(pos_adv, adv=True)
        feature_2, out_2 = net(pos_2)

        loss = criterion(out_1, out_2) + criterion(out_1, out_adv)

        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size

        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature, out = net(data.to(device, non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        if 'cifar' in dataset_name:
            feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device) 
        elif 'stl' in dataset_name:
            feature_labels = torch.tensor(memory_data_loader.dataset.labels, device=feature_bank.device) 

        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, _, target in test_bar:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            feature, out = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1).long(), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:,:1] == target.long().unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:,:5] == target.long().unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('KNN Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--root', type=str, default='../../data', help='Path to data directory')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--tau_plus', default=0.1, type=float, help='Positive class priorx')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=400, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--estimator', default='hard', type=str, help='Choose loss function')
    parser.add_argument('--dataset_name', default='stl10', type=str, help='Choose loss function')
    parser.add_argument('--beta', default=1.0, type=float, help='Choose loss function')
    parser.add_argument('--anneal', default=None, type=str, help='Beta annealing')
    parser.add_argument('--vae_path',
                        default='../results/vae_dim512_kl0.1_simclr_cifar100/model_epoch212.pth',
                        type=str, help='vae_path')
    # args parse
    args = parser.parse_args()
    feature_dim, temperature, tau_plus, k = args.feature_dim, args.temperature, args.tau_plus, args.k
    batch_size, epochs, estimator = args.batch_size, args.epochs,  args.estimator
    dataset_name = args.dataset_name
    beta = args.beta
    anneal = args.anneal

    #configuring an adaptive beta if using annealing method
    if anneal=='down':
        do_beta_anneal=True
        n_steps=9
        betas=iter(np.linspace(beta,0,n_steps))
    else:
        do_beta_anneal=False
    
    # data prepare
    train_data, memory_data, test_data = utils.get_dataset(dataset_name, root=args.root)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True, drop_last=True)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)

    # model setup and optimizer config
    model = Model(feature_dim, bn_adv_flag=True).to(device)
    model = nn.DataParallel(model)

    vae = CVAE_cifar_withbn(128, 512)
    vae.load_state_dict(torch.load(args.vae_path))
    vae.to(device)
    vae.eval()

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    c = len(memory_data.classes)
    print('# Classes: {}'.format(c))

    # training loop
    my_mkdir('../results/{}'.format(dataset_name))

    for epoch in range(1, epochs + 1):
        train_loss = train(model, vae, train_loader, optimizer, temperature, estimator, tau_plus, beta)
        
        if do_beta_anneal is True:
            if epoch % (int(epochs/n_steps)) == 0:
                beta=next(betas)

        if epoch % 100 == 0:
            test_acc_1, test_acc_5 = test(model, memory_loader, test_loader)
            if do_beta_anneal is True:
                torch.save(model.state_dict(), '../results/{}/{}_{}_model_adv_{}_{}_{}_{}_{}.pth'.format(dataset_name,dataset_name,estimator,batch_size,tau_plus,beta,epoch,anneal))
            else:
                torch.save(model.state_dict(), '../results/{}/{}_{}_model_adv_{}_{}_{}_{}.pth'.format(dataset_name,dataset_name,estimator,batch_size,tau_plus,beta,epoch))
