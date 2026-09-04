import torch.nn as nn
import torchvision.models as models


def create_resnet18(num_classes: int, pretrained: bool = True):
    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
