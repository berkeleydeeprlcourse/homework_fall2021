from typing import Union

import torch
from torch import nn

Activation = Union[str, nn.Module]


_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}


def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        activation: Activation = 'tanh',
        output_activation: Activation = 'identity',
) -> nn.Module:
    """
        Builds a feedforward neural network

        arguments:
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer

            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer

        returns:
            MLP (nn.Module)
    """
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]

    # TODO: return a MLP. This should be an instance of nn.Module
    # Note: nn.Sequential is an instance of nn.Module.
    # raise NotImplementedError

    # build the sequential model
    model = nn.Sequential()

    # 1. add input layer
    model.add_module('input', nn.Linear(input_size, size))

    # 2. add hidden layers
    for n in range(n_layers):
        model.add_module('hidden'+str(n), nn.Linear(size, size))
        model.add_module('activation'+str(n), activation)

    # 3. add output layer
    model.add_module('output', nn.Linear(size, output_size))
    model.add_module('out activation', output_activation)

    # 4. assert nn.Sequential model is a nn.Module
    assert(isinstance(model, nn.Module))

    return model

    '''
    Another method is to use list = [('name', nn.Linear()), (), ()] , then nn.Sequential(collections.OrderedDict(list))
    Or list = [nn.Linear(), xx ,xx] and nn.Sequential(list)

    '''

device = None


def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()
