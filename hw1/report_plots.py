import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import os

# manually add data of Ant and Halfcheeta


def plot_mean_std(ax, iterations, mean, std, mean_expert, mean_bc):
    mean = np.array(mean)
    std = np.array(std)

    ax.plot(iterations, mean_expert, 'r', label='expert')
    ax.plot(iterations, mean_bc, 'g', label='naive bc')
    ax.plot(iterations, mean, 'b-s', label='DAgger mean')
    ax.fill_between(iterations, mean-std, mean+std, alpha=0.2, label='DAgger std')


def set_plot_env(iterations, mean, std, mean_expert, mean_bc, exp_name):

    plt.figure(figsize=(10,5))
    style = "whitegrid"
    sns.set_theme(style=style) # background color
    ax = plt.gca()
    plot_mean_std(ax, iterations, mean, std, mean_expert, mean_bc)

    ax.legend()
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Return')
    ax.set_title('return of exp ' + exp_name)

    exp_dir = 'plots/'
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    plt.savefig(fname=exp_dir + exp_name + '.svg', format='svg')

def plot_DAgger(mean, std, exp_name):
    iterations = np.arange(len(mean))
    I = np.ones(len(mean))
    mean_bc = mean[0] * I 
    mean_expert = mean[-1] * I 
    set_plot_env(iterations, mean, std, mean_expert, mean_bc, exp_name)

class Ant_exp:
    mean = [4274., 4648., 4746., 4619., 4447., 4356., 4731., 4739., 4581., 4834., 4648., 4382., 4750., 4809., 4729., 4718., 4492., 4829., 4676., 4682.]
    std =  [1128.,   53.,   85.,  103.,  854., 1030.,  124.,  135.,  336.,  109.,  499., 1152.,  91.,   68.,  104.,   72.,  169.,  127.,   81.,  375.]

    ep_len = [100, 300, 500, 800, 1000, 1500, 2000]
    mean_len = [384, ]
    std_len = [66, ]

if __name__ == "__main__":
    # no_iter = np.arange(10)
    # mean = np.linspace(0.9, 1.1, 10)
    # std = np.linspace(0, 0.2, 10)

    # mean_expert = np.ones(10) * 1.3
    # mean_bc = np.ones(10) * 0.9

    exp = Ant_exp
    mean = exp.mean
    std = exp.std


    plot_DAgger(mean, std, 'Ant')