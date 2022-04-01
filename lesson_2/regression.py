import pandas as pd
import torch
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def linear_regression():
    path = '../data.csv'
    data = pd.read_csv(path).to_numpy()
    data = torch.Tensor(data)
    x = data[:, :2]
    y = data[:, 2]
    x = torch.cat([torch.ones(x.size(0), 1), x], 1)
    w = (x.t().mm(x).inverse()).mm(x.t().mm(y.unsqueeze(1)))

    xx1 = torch.linspace(0, x[:, 1].max().item()*1.1, 100)
    xx = torch.cat([torch.ones(xx1.size(0), 1), xx1.unsqueeze(1), torch.ones(xx1.size(0), 1)*3], 1)
    yy = xx.mm(w)

    plt.title("housing prices")
    plt.xlabel("square feet")
    plt.ylabel("price")
    plt.plot(xx1.numpy(), yy.numpy(), '-', x[:, 1].numpy(), y.numpy(), '.')
    plt.show()


def polynomial_regression():
    path = '../data.csv'
    data = pd.read_csv(path).to_numpy()
    data = torch.Tensor(data)
    x = data[:, :2]
    y = data[:, 2]
    x = torch.cat([torch.ones(x.size(0), 1), x, x[:, 0].pow(2).unsqueeze(1), x[:, 0].pow(3).unsqueeze(1)], 1)
    w = (x.t().mm(x).inverse()).mm(x.t().mm(y.unsqueeze(1)))

    xx1 = torch.linspace(0, x[:, 1].max().item()*1.1, 100).unsqueeze(1)
    rooms = torch.ones(xx1.size(0), 1) * 3
    rooms[xx1 < 1000] = 2
    rooms[xx1 > 4000] = 4
    xx = torch.cat([torch.ones(xx1.size(0), 1), xx1, rooms,
                    xx1.pow(2), xx1.pow(3)], 1)
    yy = xx.mm(w)

    plt.title("housing prices")
    plt.xlabel("square feet")
    plt.ylabel("price")
    plt.plot(xx1.numpy(), yy.numpy(), '-', x[:, 1].numpy(), y.numpy(), '.')
    plt.show()


if __name__ == '__main__':
    linear_regression()
    polynomial_regression()
