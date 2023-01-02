from torch import nn
from torch.nn import functional as F
import torch
import numpy as np
from torchvision import models
import torchvision 

NUM_CLASSES = 13
class Net_2(nn.Module):

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
        self.fc2 = nn.Linear(fc_neurons, NUM_CLASSES)
        
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)


    def forward(self, x):
        """
        Forward pass of the network

        Args:
            x: batch of images tensor(n, 3, 128, 128)

        Returns:
            logits of the images tensor(n, 13)
        """        
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

    def features(self, x):
        """
        Extracts the feature maps using the 
        convolutional layers
        """
        x = self.conv1(x)
        x = self.pool(self.activation(self.bn1(x)))

        x = self.conv2(x)
        x = self.pool(self.activation(self.bn2(x)))
        
        x = self.conv3(x)
        x = self.pool(self.activation(self.bn3(x)))

        return x


class FineTunedEffnet(nn.Module):

    # The normalization is done here to 
    # keep the dataloader the same for both
    # models
    normalize = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], # normalization of ImageNet
        std=[0.229, 0.224, 0.225]
    )

    # Default weights, trained on ImageNet
    model_weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT

    def __init__(self):
        super(FineTunedEffnet, self).__init__()
        self.effnet = models.efficientnet_b0(weights=self.model_weights)
        self.effnet.classifier[1] = nn.Linear(1280, NUM_CLASSES)

    def forward(self, x):
        # Normalization before passing to the model
        x = self.normalize(x)
        y = self.effnet(x)
        return y

