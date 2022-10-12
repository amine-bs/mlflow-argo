import torch.nn as nn
from torchvision import models

class ResNet(nn.Module):

  def __init__(self, class_num=2, architecture="resnet18", pretrained=True):
    super(ResNet,self).__init__()
    self.pretrained = pretrained
    model = models.resnet18(pretrained=pretrained)
    #change the dimension of output
    fc_input_dim = model.fc.in_features
    model.fc = nn.Linear(fc_input_dim, class_num)
    self.model = model
    
  def forward(self, x):
    x = self.model(x)
    return x

