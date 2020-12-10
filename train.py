"""
train Energy based model
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import argparse

from models import mlp 
from langevin import sample_langevin
from data import sample_data

def train(hparams):
    
    model = mlp(sizes=[2, 128, 128, 1], activation=nn.ReLU)
    optimizer = Adam(model.parameters(), lr=hparams.lr)
    
    # load dataset
    N_train = 5000
    N_val = 1000
    N_test = 5000

    X_train = sample_data(N_train)
    X_val = sample_data(N_val)
    X_test = sample_data(N_test)

    train_dl = DataLoader(X_train, batch_size=32, shuffle=True, num_workers=8)
    val_dl = DataLoader(X_val, batch_size=32, shuffle=True, num_workers=8)
    test_dl = DataLoader(X_test, batch_size=32, shuffle=True, num_workers=8)

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

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()
            losses.append(loss.item())

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate. default: 1e-3')
    parser.add_argument('--stepsize', type=float, default=0.1, help='Langevin dynamics step size')
    parser.add_argument('--n_steps', type=int, default=100, help='The number of Langevin dynamics steps')
    parser.add_argument('--n_epochs', type=int, default=100, help='The number of training epoches')
    parser.add_argument('--alpha', type=float, default=1.0, help='Regularizer coefficient for contrastive divergence')
    hparams = parser.parse_args()

    train(hparams)
