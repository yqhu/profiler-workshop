# Based on: https://github.com/pytorch/examples/blob/master/mnist/main.py
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import time


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        with torch.profiler.record_function("conv1"):
            x = self.conv1(x)
            x = self.bn1(x)
            x = F.relu(x)
        with torch.profiler.record_function("conv2"):
            x = self.conv2(x)
            x = self.bn2(x)
            x = F.relu(x)
        with torch.profiler.record_function("head"):
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            output = F.log_softmax(x, dim=1)
        return output


def test(model, test_loader):
    with torch.inference_mode():
        for data, target in test_loader:
            output = model(data)


def main():
    test_kwargs = {'batch_size': 64, 'num_workers': 4}

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    test_dataset = datasets.MNIST('../data', train=False,
                       transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    # eager mode
    model = Net().eval()
    test(model, test_loader)

    start = time.perf_counter()
    test(model, test_loader)
    print(f'eager mode: {1000 * (time.perf_counter() - start):.1f} ms')

    # jit
    model = Net().eval()
    model = torch.jit.script(model)
    
    test(model, test_loader)
    
    start = time.perf_counter()
    test(model, test_loader)
    print(f'jit: {1000 * (time.perf_counter() - start):.1f} ms')

    # ofi
    model = Net().eval()
    model = torch.jit.script(model)
    model = torch.jit.optimize_for_inference(model.eval())

    test(model, test_loader)

    start = time.perf_counter()
    test(model, test_loader)
    print(f'ofi: {1000 * (time.perf_counter() - start):.1f} ms')


if __name__ == '__main__':
    main()
