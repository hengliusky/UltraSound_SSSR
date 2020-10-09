import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Discriminator(nn.Module):
    def __init__(self, conv_dim=64, repeat_num=2):
        super(Discriminator, self).__init__()
        layers = []
        # net0
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2))

        curr_dim = conv_dim
        # net1,2
        for i in range(0, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.BatchNorm2d(curr_dim * 2, affine=True, track_running_stats=True))
            curr_dim = curr_dim * 2

        # net3
        layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=1, padding=1))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.BatchNorm2d(curr_dim * 2, affine=True, track_running_stats=True))
        curr_dim = curr_dim * 2

        # net4
        layers.append(nn.Conv2d(curr_dim, curr_dim, kernel_size=4, stride=1, padding=1))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.BatchNorm2d(curr_dim, affine=True, track_running_stats=True))
        # net5
        layers.append(nn.Conv2d(curr_dim, 1, kernel_size=4, stride=1, padding=1))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        res = torch.sigmoid(logits)
        return logits


if __name__ == "__main__":
    #x = np.random.random(1, 3, 64, 64)
    x = torch.randn(1, 3, 64, 64)
    D = Discriminator()
    logits = D(x)
    print(logits.shape)
   