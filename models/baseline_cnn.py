import torch.nn as nn
import torch.nn.functional as F

class BaselineCNN(nn.Module):
    # TODO: what are best possible parameters
    def __init__(self, maxpool_kernel_size=4, out1=6, kernel1=5, in2=6, out2=16, kernel2=5):
        super(BaselineCNN, self).__init__()
        self.out2 = out2
        self.conv1 = nn.Conv2d(3, out2, kernel1)
        self.pool = nn.MaxPool2d(maxpool_kernel_size, maxpool_kernel_size)
        self.conv2 = nn.Conv2d(in2, out2, kernel2)
        self.fc1 = nn.Linear(self.out2 * 61 * 61, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 200)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.out2 * 61 * 61)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
