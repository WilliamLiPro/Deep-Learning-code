import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def show_activation(act: nn.Module):
    x = torch.linspace(-6, 6, 100)
    x.requires_grad_()
    y = act(x)
    loss = y.sum()
    g = torch.autograd.grad(loss, x)[0]

    plt.plot(x.data.numpy(), y.data.numpy(), label=act.__class__.__name__)
    plt.plot(x.data.numpy(), g.data.numpy(), label='gradient')
    plt.legend()
    plt.show()


def show_activations():
    act_f_s = [nn.Sigmoid, nn.Hardsigmoid, nn.Tanh, nn.Hardswish,
               nn.Softsign, nn.ReLU, nn.Softplus, nn.LeakyReLU,
               nn.ELU, nn.RReLU]
    for act_f in act_f_s:
        act = act_f().train()
        show_activation(act)


def show_softmax():
    x = torch.linspace(-6, 6, 100)
    sz = x.size()[0]
    x1 = x.unsqueeze(1).expand((sz, sz))
    x2 = x.unsqueeze(0).expand((sz, sz))
    xx = torch.cat([x1.unsqueeze(2), x2.unsqueeze(2)], 2)

    act = nn.Softmax(dim=2)
    y = act(xx)
    y1 = y[:, :, 0]
    y2 = y[:, :, 1]

    plt.imshow(y1.data.numpy(), label=act.__class__.__name__)
    plt.colorbar()
    plt.xticks(np.arange(0, sz, 99), x.data[0::99].numpy())
    plt.yticks(np.arange(0, sz, 99), x.data[0::99].numpy())
    plt.show()

    plt.imshow(y2.data.numpy(), label=act.__class__.__name__)
    plt.colorbar()
    plt.xticks(np.arange(0, sz, 99), x.data[0::99].numpy())
    plt.yticks(np.arange(0, sz, 99), x.data[0::99].numpy())
    plt.show()


if __name__ == '__main__':
    show_activations()
    show_softmax()
