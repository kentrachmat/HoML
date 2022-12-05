from torch import nn
from torch.nn import functional as F


class Net_2(nn.Module):
    NUM_CLASSES = 13

    def __init__(self, fc_neurons, channels):
        super(Net_2, self).__init__()
        [X,Y,Z] = channels

        self.conv1 = nn.Conv2d(3, X, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(X, Y, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(Y, Z, kernel_size=7, padding=2)
        self.bn1 = nn.BatchNorm2d(X)
        self.bn2 = nn.BatchNorm2d(Y)
        self.bn3 = nn.BatchNorm2d(Z)
        self.fc1 = nn.Linear(Z,fc_neurons)
        self.fc2 = nn.Linear(fc_neurons, self.NUM_CLASSES)
        
        self.activation = nn.ReLU()
        self.drop = nn.Dropout(p=0.25)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)


    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(self.activation(self.bn1(x)))

        x = self.conv2(x)
        x = self.pool(self.activation(self.bn2(x)))
        
        x = self.conv3(x)
        x = self.pool(self.activation(self.bn3(x)))

        x = F.adaptive_avg_pool2d(x,  (1, 1))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = self.activation(self.fc1(x))
        x = self.fc2(x)

        return x


