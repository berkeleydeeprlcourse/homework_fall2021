import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import os


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

    ax.legend(loc='center right')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Return')
    ax.set_title('return of ' + exp_name +' experiment')
    ax.set_xlim([-0.5,10])

    exp_dir = 'plots/'
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    plt.savefig(fname=exp_dir + 'figure-2_' + exp_name + '.png', format='png')

def plot_DAgger(mean, std, mean_expert, exp_name):
    iterations = np.arange(len(mean))
    I = np.ones(len(mean))
    mean_bc = mean[0] * I 
    mean_expert = mean_expert * I 
    set_plot_env(iterations, mean, std, mean_expert, mean_bc, exp_name)

def plot_changing_ep(ep_len, mean_len, std_len):
    plt.figure(figsize=(10,5))
    style = "whitegrid"
    sns.set_theme(style=style) # background color
    ax = plt.gca()

    mean_len = np.array(mean_len)
    std_len = np.array(std_len)
    I = np.ones(len(mean_len))
    mean_expert = Ant_exp.mean_expert * I 

    ax.plot(ep_len, mean_expert, 'r', label='expert')
    plt.plot(ep_len, mean_len, 'b-s', label='bc mean')
    plt.fill_between(ep_len, mean_len-std_len, mean_len+std_len, alpha=0.2, label='bc std')
    ax.legend(loc='upper left')
    ax.set_xlabel('num of traning steps')
    ax.set_ylabel('Return')
    ax.set_title('return of ' + 'Ant experiments with varying traning steps')

    exp_dir = 'plots/'
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    plt.savefig(fname=exp_dir + 'figure-1_varying_train_step' + '.png', format='png')


class Ant_exp:
    mean = [4274., 4648., 4746., 4619., 4447., 4356., 4731., 4739., 4581., 4834.]
    std =  [1128.,   53.,   85.,  103.,  854., 1030.,  124.,  135.,  336.,  109.]
    mean_expert = 4710

    ep_len = [100, 300, 500, 700, 990, 1100, 1400, 1500, 1700]
    mean_len = [567, 1505, 3849, 3296, 3774, 3570, 2227, 4236, 4249]
    std_len =  [7,   1250, 1299, 1604, 1363, 1392, 1740, 1151, 901]

class Hooper_exp:
    mean = [ 523., 1700., 2453., 3763., 3778., 3790., 3388., 3771., 3537., 3788.]
    std  = [ 75., 607., 725.,   4.,   3.,   4., 671.,   3., 160.,   3.]
    mean_expert = 3779
    

if __name__ == "__main__":

    # figure 1
    exp = Ant_exp
    mean_len = exp.mean_len
    std_len = exp.std_len
    ep_len = exp.ep_len
    plot_changing_ep(ep_len, mean_len, std_len)

    # figure 2-1
    exp = Ant_exp
    mean = exp.mean
    std = exp.std
    mean_expert = exp.mean_expert
    plot_DAgger(mean, std, mean_expert, 'Ant')

    # figure 2-2
    exp = Hooper_exp
    mean = exp.mean
    std = exp.std
    mean_expert = exp.mean_expert
    plot_DAgger(mean, std, mean_expert, 'Hooper')