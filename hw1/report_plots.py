import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

# manually add data of Ant and Halfcheeta


no_iter = np.arange(10)
mean = []
std = []
mean_expert = np.ones(10) * 1
mean_bc = np.ones(10) * 1

sns.set_theme(style="darkgrid")

# Load an example dataset with long-form data
fmri = sns.load_dataset("fmri")

# Plot the responses for different events and regions
sns.lineplot(x="timepoint", y="signal",
             hue="region", style="event",
             data=fmri)