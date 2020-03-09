import torch.nn as nn
import torch.nn.functional as F

class BaselineCNN(nn.Module):
    def __init__(self, maxpool_kernel_size=3, out1=8, kernel1=6, out2=20, kernel2=6, grey=False):
        super(BaselineCNN, self).__init__()
        self.out2 = out2
        self.kernel2 = kernel2
        in_size = 1 if grey else 3
        self.flattened_size = 20*26*26
        self.conv1 = nn.Conv2d(in_size, out1, kernel1)
        self.bn2d = nn.BatchNorm2d(out1)
        self.pool = nn.MaxPool2d(maxpool_kernel_size, maxpool_kernel_size)
        self.conv2 = nn.Conv2d(out1, out2, kernel2)
        self.fc1 = nn.Linear(self.flattened_size, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 200)
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn2d(self.conv1(x))))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.flattened_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.drop(self.fc2(x)))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
