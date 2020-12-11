"""
train energy based model
"""
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import argparse
import os
import random
from utils import seed_everything

from models import mlp
from langevin import sample_langevin
from data import sample_data
import wandb

def train(hparams):
    #wandb.init(project="ebm-gaussians")

    seed_everything(hparams.seed)
    model = mlp(sizes=[2, 100, 100, 1], activation=nn.ReLU)
    optimizer = Adam(model.parameters(), lr=hparams.lr)

    # load dataset
    N_train = 5000

    X_train = sample_data(N_train)

    train_dl = DataLoader(X_train, batch_size=100, shuffle=True, num_workers=8)
    losses = []

    for _ in range(hparams.n_epochs):
        for x in train_dl:

            neg_x = torch.randn_like(x)
            neg_x = sample_langevin(neg_x, model, hparams.stepsize, hparams.n_steps)

            optimizer.zero_grad()

            pos_out = model(x)
            neg_out = model(neg_x)

            loss = (pos_out - neg_out) + hparams.alpha * (pos_out ** 2 + neg_out ** 2)

            loss = loss.mean()
            loss.backward()

            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()

            losses.append(loss.item())
            # wandb.log({'loss': loss.item()})

    print('saving a trained model')
    torch.save(model, hparams.model_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=float, default=42,
                        help='seed')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate. default: 1e-3')
    parser.add_argument('--stepsize', type=float, default=0.01,
                        help='Langevin dynamics step size')
    parser.add_argument('--n_steps', type=int, default=100,
                        help='The number of Langevin dynamics steps')
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='The number of training epoches')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Regularizer coefficient for contrastive divergence')
    parser.add_argument('--model_path', type=str, default='Models/gbm_model.pt',
                        help='trained model save path')

    hparams = parser.parse_args()

    train(hparams)
