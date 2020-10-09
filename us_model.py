import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#encoder
class Encoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Encoder,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            #nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            #nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(True)
        )

    def forward(self, input):
        return self.conv(input)

#decoder
class Decoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Decoder,self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 3, padding=1),
            #nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            #nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(True)
        )
    def forward(self, input):
        return self.conv(input)


class USGan_g(nn.Module):
    def __init__(self):
        super(USGan_g, self).__init__()
        self.encoder1 = Encoder(3, 32)
        self.decoder1 = Decoder(32, 32)
        self.encoder2 = Encoder(32, 32)
        self.decoder2_1 = Decoder(32, 32)
        self.decoder2_2 = Decoder(32, 32)
        self.encoder3 = Encoder(32, 64)
        self.decoder3_1 = Decoder(64, 64)
        self.decoder3_2 = Decoder(64, 32)
        self.decoder3_3 = Decoder(32, 32)
        self.conv = nn.Conv2d(32, 3, 3, stride=1, padding=1)
        self.final_conv = nn.Conv2d(9, 3, 3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        tmp1 = self.encoder1(x)
        out1 = self.conv(self.decoder1(tmp1))
        rec1 = out1 + x
        tmp2 = self.encoder2(tmp1)
        out2 = self.conv(self.decoder2_2(self.decoder2_1(tmp2)))
        rec2 = out2 + x
        tmp3 = self.encoder3(tmp2)
        out3 = self.conv(self.decoder3_3(self.decoder3_2(self.decoder3_1(tmp3))))
        rec3 = out3 + x
        outputs = torch.cat([rec1, rec2, rec3], dim=1)
        outputs = self.final_conv(outputs)
        outputs = self.tanh(outputs)
        return torch.clamp(outputs, min=0, max=1)


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
        # res = torch.sigmoid(logits)
        return logits



def low2high_test():
    net = USGan_g().cuda()
    x = np.random.randn(1, 3, 64, 64).astype(np.float32)
    x = torch.from_numpy(x).cuda()
    y = net(x)
    print(y)






