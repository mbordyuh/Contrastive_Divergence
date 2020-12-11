"""
test Energy based model
"""
import torch
from torch.utils.data import DataLoader
import argparse
import matplotlib.pyplot as plt
from data import sample_data
import seaborn as sns

from langevin import sample_langevin

def plot_figure(x_start, x_end, save_path):

    sns.set_style('white')
    plt.figure(figsize=(7, 5), dpi=100)
    plt.plot(x_start.detach()[:, 0], x_start.detach()[:, 1], 'o', markersize=0.5)
    plt.plot(x_end.detach()[:, 0], x_end.detach()[:, 1], 'o', markersize=0.5)
    plt.legend(['Initial distribution', 'Final distribution'])
    plt.savefig(save_path)

def test(hparams):
    # load dataset
    N_test = 5000
    model = torch.load(hparams.model_path)

    x_start = torch.randn([N_test, 2])
    x_end = sample_langevin(x_start, model, hparams.stepsize, hparams.n_steps)


    plot_figure(x_start, x_end, save_path='Figures/ebm.pdf')




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate. default: 1e-3')
    parser.add_argument('--stepsize',type=float, default=0.1,
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

    test(hparams)
