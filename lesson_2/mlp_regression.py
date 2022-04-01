import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from torch.optim import SGD, Adam


class TinyMlp(nn.Module):
    def __init__(self, nodes, x):
        super(TinyMlp, self).__init__()
        self.x_mean = x.mean(dim=0)
        self.x_std = x.std(dim=0)
        n_nodes = len(nodes)
        layers = []
        # layers
        for i in range(n_nodes - 2):
            node1 = nodes[i]
            node2 = nodes[i + 1]
            layers.append(nn.Linear(node1, node2))
            layers.append(nn.Sigmoid())
        # common layers
        layers.append(nn.Linear(nodes[-2], nodes[-1]))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        x = (x - self.x_mean)/self.x_std
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x


def trainer(model: nn.Module, optim: torch.optim.Optimizer,
            data_path: str='../data.csv', epoch=1000):
    data = pd.read_csv(data_path).to_numpy()
    data = torch.Tensor(data)
    x = data[:, :2]
    y = data[:, 2]
    y_mean = y.mean()
    y_std = y.std()
    y = (y-y_mean)/y_std

    criterion = nn.MSELoss(reduce=True, size_average=True)

    for epo in range(epoch):
        model.train()
        output = model(x)
        loss = criterion(output[:, 0], y)

        loss.backward()

        optim.step()
        model.zero_grad()
        model.eval()


def mlp_regression():
    # data
    path = '../data.csv'
    data = pd.read_csv(path).to_numpy()
    data = torch.Tensor(data)
    x = data[:, :2]
    y = data[:, 2]
    y_mean = y.mean()
    y_std = y.std()

    # training
    net = TinyMlp([2, 1024, 512, 1], x)
    optim = SGD(net.parameters(), lr=0.01, momentum=0.9)
    trainer(net, optim)
    optim = SGD(net.parameters(), lr=0.001, momentum=0.9)
    trainer(net, optim)

    # output
    xx1 = torch.linspace(500, x[:, 0].max().item() * 1.1, 100).unsqueeze(1)
    rooms = torch.ones(xx1.size()[0], 1) * 3
    xx = torch.cat([xx1, rooms], 1)

    net.eval()
    yy = net(xx).data * y_std + y_mean

    plt.title("housing prices")
    plt.xlabel("square feet")
    plt.ylabel("price")
    plt.plot(xx1.numpy(), yy.numpy(), '-', x[:, 0].numpy(), y.numpy(), '.')
    plt.show()


if __name__ == '__main__':
    mlp_regression()
