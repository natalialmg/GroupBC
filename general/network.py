
from torchvision import models
import torch.nn as nn


## Resnet network
def get_resnet(nclasses, pretrained = False, typenet = 'resnet34', convert_conv1 = True):

    if typenet == 'resnet18':
        model = models.resnet18(pretrained=pretrained)

    elif typenet == 'resnet34':
        model = models.resnet34(pretrained=pretrained)

    if convert_conv1:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.maxpool = nn.Identity()

    # model.maxpool = nn.Identity()
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=nclasses)

    return model