import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import numpy as np

class Model(nn.Module):
    def __init__(self, model):
        super(Model, self).__init__()
        if model=="resnet50":
            self.model = models.resnet50()
        elif model=="resnet101":
            self.model = models.resnet101()
        elif model=="resnet152":
            self.model = models.resnet152()
        else:
            self.model = models.resnet50()

    def forward(self, x):
        return self.model(x)