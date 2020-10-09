import torch
import numpy as np
from torch.autograd import Variable
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt


# 根据vgg19自定义一个网络用来提取不同层的输出特征
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        self.model = models.vgg19(pretrained=True).features[:28]

        #if not requires_grad:
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.model(x)
        return out


if __name__ == "__main__":
    net = Vgg19().cuda()
    X = np.random.randn(32, 3, 224, 224).astype(np.float32)
    X = torch.from_numpy(X).cuda()
    y = net(X)
    print(y.shape)
