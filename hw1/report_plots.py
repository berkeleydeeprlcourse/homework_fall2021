import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import os

# manually add data of Ant and Halfcheeta


def plot_mean_std(ax, iterations, mean, std, mean_exp, mean_bc):
    

    ax.plot(iterations, mean_expert, 'r', label='expert')
    ax.plot(iterations, mean_bc, 'g', label='naive bc')
    ax.plot(iterations, mean, 'b-s', label='DAgger mean')
    ax.fill_between(iterations, mean-std, mean+std, alpha=0.2, label='DAgger std')
    


def set_plot_env(iterations, mean, std, mean_exp, mean_bc, exp_name):

    plt.figure(figsize=(10,5))
    style = "whitegrid"
    sns.set_theme(style=style) # background color
    ax = plt.gca()
    plot_mean_std(ax, no_iter, mean, std, mean_expert, mean_bc)

    ax.legend()
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Return')
    ax.set_title('return of exp')

    exp_dir = 'plots/'
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    plt.savefig(fname=exp_dir + exp_name + '.svg', format='svg')


class Ant_exp:
    mean = [1]
    std =  []
    mean_bc = mean[0]
    mean_expert = mean[-1]
    iterations = np.arange(len(mean))

if __name__ == "__main__":
    no_iter = np.arange(10)
    mean = np.linspace(0.9, 1.1, 10)
    std = np.linspace(0, 0.2, 10)

    mean_expert = np.ones(10) * 1.3
    mean_bc = np.ones(10) * 0.9

    set_plot_env(no_iter, mean, std, mean_expert, mean_bc, 'Ant')