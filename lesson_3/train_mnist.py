import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim import SGD, Adam
from lesson_3 import simple_trainer, save_csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TinyMlp(nn.Module):
    def __init__(self, nodes):
        super(TinyMlp, self).__init__()
        n_nodes = len(nodes)
        layers = []
        # Rosler layers
        for i in range(n_nodes - 2):
            node1 = nodes[i]
            node2 = nodes[i + 1]
            layers.append(nn.Linear(node1, node2))
            layers.append(nn.ReLU())
        # common layers
        layers.append(nn.Linear(nodes[-2], nodes[-1]))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.layers(x)
        return x


class TinyCnn(nn.Module):
    def __init__(self, layer_channels, classes, kernel_size=3,
                 stride=1, padding=1, dilation=1, bias=False, padding_mode='zeros'):
        super().__init__()
        n_layers = len(layer_channels)
        conv_layers = []

        # conv layers
        for i in range(n_layers-1):
            in_channels = layer_channels[i]
            channel_out = layer_channels[i+1]
            conv_layers.append(nn.Conv2d(in_channels, channel_out, kernel_size,
                                         stride, padding, dilation, bias=bias, padding_mode=padding_mode))
            conv_layers.append(nn.ReLU())
            if i == 1:
                conv_layers.append(nn.MaxPool2d((2, 2)))
        self.conv_layers = nn.Sequential(*conv_layers)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(layer_channels[n_layers-1], classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x


def mnist_trainer(model: nn.Module, opt: torch.optim.Optimizer, criterion: nn.Module,
                  mnist_path: str = 'E:/Dataset/Object_Recognition/',
                  save_path: str = '../result', epoch=50, batch_size=512,
                  device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                  multi_gpu=True):
    parameter_n = sum([param.nelement() for param in model.parameters()])
    save_name = 'MNIST_{} model_{} param_{}'.format(opt.__class__.__name__,
                                                    model.__class__.__name__, parameter_n)

    transform_train = transforms.Compose(
        [transforms.RandomCrop(28, padding=4, padding_mode='reflect'),
         transforms.ToTensor(),
         transforms.Normalize((0.13,), (0.31,)),])

    transform_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.13,), (0.31,))])

    train_set = datasets.MNIST(root=mnist_path, train=True, download=False, transform=transform_train)
    test_set = datasets.MNIST(root=mnist_path, train=False, download=False, transform=transform_test)
    train_loader = data.DataLoader(train_set, batch_size=batch_size, pin_memory=False, shuffle=True, num_workers=8,
                                   drop_last=True)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, pin_memory=False, shuffle=False, num_workers=8)

    if multi_gpu and device.type == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print('using {} GPUs'.format(torch.cuda.device_count()))
    model = model.to(device)

    summaries = simple_trainer(model, criterion, opt, [train_loader, test_loader],
                               epochs=epoch, save_path=save_path + '/checkpoint/' + save_name + '.pth',
                               device=device)

    save_csv(summaries, save_path+'/summary/' + save_name)
    return summaries


def mnist_mlp_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mnist_path = 'E:/Dataset/Object_Recognition/'
    # mnist_path = '/home/lwp/ImgData/Dataset/image_recognition/'
    epoch = 50
    batch_size = 512

    # MLP
    net = TinyMlp([784, 1024, 512, 256, 10]).to(device)
    opt = SGD(net.parameters(), lr=0.001, momentum=0.9, nesterov=True)
    criterion = nn.CrossEntropyLoss(reduce=False)
    summaries = mnist_trainer(net, opt, criterion, mnist_path, './result',
                        epoch=epoch, batch_size=batch_size,
                        device=device, multi_gpu=False)

    net = TinyMlp([784, 1024, 512, 256, 10]).to(device)
    opt = Adam(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(reduce=False)
    summaries = mnist_trainer(net, opt, criterion, mnist_path, './result',
                        epoch=epoch, batch_size=batch_size,
                        device=device, multi_gpu=False)


def mnist_cnn_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # mnist_path = 'E:/Dataset/Object_Recognition/'
    mnist_path = '/home/lwp/ImgData/Dataset/image_recognition/'
    epoch = 50
    batch_size = 512

    # CNN
    net = TinyCnn([32, 64, 128, 128], 10).to(device)
    opt = SGD(net.parameters(), lr=0.001, momentum=0.9, nesterov=True)
    criterion = nn.CrossEntropyLoss(reduce=False)
    summaries = mnist_trainer(net, opt, criterion, mnist_path, './result',
                              epoch=epoch, batch_size=batch_size,
                              device=device, multi_gpu=False)

    net = TinyCnn([32, 64, 128, 128], 10).to(device)
    opt = Adam(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(reduce=False)
    summaries = mnist_trainer(net, opt, criterion, mnist_path, './result',
                              epoch=epoch, batch_size=batch_size,
                              device=device, multi_gpu=False)


if __name__ == '__main__':
    mnist_mlp_test()
    mnist_cnn_test()
