import matplotlib.pyplot as plt

# data = {1: 1433.59, 2: 1638.22, 3: 3193.89, 4: 2702.08, 5: 1499.70}

# xs = data.keys()
# ys = data.values()
# plt.plot(xs, ys)
# plt.title('Policy Performance over Number of Layers on Walker2d Dataset')
# plt.xlabel('n_layers')
# plt.ylabel('eval_average_return')
# plt.show()

data_dagger_walker2d = {
    0: 1638.23,
    1: 5335.63,
    2: 5367.04,
    3: 4516.48,
    4: 5493.41,
    5: 5489.95,
    6: 5550.95,
    7: 5566.29,
    8: 5468.39,
    9: 5551.07
}

data_dagger_ant = {
    0: 3399.31,
    1: 4441.09,
    2: 4506.26,
    3: 4666.11,
    4: 4711.38,
    5: 4617.09,
    6: 4705.11,
    7: 4766.76,
    8: 4664.41,
    9: 4781.87
}

xs = data_dagger_ant.keys()
ys = data_dagger_ant.values()
plt.plot(xs, ys)
plt.title('DAgger Performance over Iteration Number on Ant Dataset')
plt.xlabel('iter')
plt.ylabel('eval_average_return')
plt.show()