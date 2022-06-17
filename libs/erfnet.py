# ERFNet full model definition for Pytorch
# Sept 2017
# Eduardo Romera
#######################

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.Conv2d(ninput,
                              noutput - ninput, (3, 3),
                              stride=2,
                              padding=1,
                              bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output, inplace=True)


class FuseBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv2d_1_1 = nn.Conv2d(ninput, noutput, 1, 1)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, x):
        x = self.conv2d_1_1(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class non_bottleneck_1d(nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann,
                                   chann, (3, 1),
                                   stride=1,
                                   padding=(1, 0),
                                   bias=True)

        self.conv1x3_1 = nn.Conv2d(chann,
                                   chann, (1, 3),
                                   stride=1,
                                   padding=(0, 1),
                                   bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann,
                                   chann, (3, 1),
                                   stride=1,
                                   padding=(1 * dilated, 0),
                                   bias=True,
                                   dilation=(dilated, 1))

        self.conv1x3_2 = nn.Conv2d(chann,
                                   chann, (1, 3),
                                   stride=1,
                                   padding=(0, 1 * dilated),
                                   bias=True,
                                   dilation=(1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):

        output = self.conv3x1_1(input)
        output = F.relu(output, inplace=True)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output, inplace=True)

        output = self.conv3x1_2(output)
        output = F.relu(output, inplace=True)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        return F.relu(output + input,
                      inplace=True)  # +input = identity (residual connection)


class Encoder(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.initial_block = DownsamplerBlock(in_channel, 32)
        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(32, 64))

        for _ in range(0, 5):  # 5 times
            self.layers.append(non_bottleneck_1d(64, 0.03, 1))

        self.layers.append(DownsamplerBlock(64, 128))

        for _ in range(0, 2):  # 2 times
            self.layers.append(non_bottleneck_1d(128, 0.3, 2))
            self.layers.append(non_bottleneck_1d(128, 0.3, 4))
            self.layers.append(non_bottleneck_1d(128, 0.3, 8))
            self.layers.append(non_bottleneck_1d(128, 0.3, 16))

    def forward(self, input):
        masklist = []
        output = self.initial_block(input)
        masklist.append(output)
        for i, layer in enumerate(self.layers):
            output = layer(output)
            if i == 5:
                masklist.append(output)
            elif i == len(self.layers) - 1:
                masklist.append(output)
            else:
                continue
        return masklist


class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput,
                                       noutput,
                                       3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1,
                                       bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output, inplace=True)


class UDsampling(nn.Module):
    def __init__(self, ninput, noutput, up_factor=4):
        super().__init__()
        self.upin = nn.UpsamplingBilinear2d(scale_factor=up_factor)
        self.conv = nn.Sequential(nn.Conv2d(ninput, noutput, 1, 1),
                                  nn.BatchNorm2d(noutput, eps=1e-3),
                                  nn.ReLU(inplace=True))
        self.conv_1 = non_bottleneck_1d(noutput, 0, 1)

    def forward(self, x):
        up_x = self.upin(x)
        up_x = self.conv(up_x)
        return self.conv_1(up_x)


class Decoder(nn.Module):
    def __init__(self, out_channel):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(UpsamplerBlock(256, 64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(UpsamplerBlock(128, 32))
        self.layers.append(non_bottleneck_1d(32, 0, 1))
        self.layers.append(non_bottleneck_1d(32, 0, 1))
        self.conv_out = nn.ConvTranspose2d(64,
                                           out_channel,
                                           3,
                                           stride=2,
                                           padding=1,
                                           output_padding=1,
                                           bias=True)

        #self.fuse = nn.Softmax(dim=1)

    def forward(self, inputlist, pflist):
        x_1 = self.layers[0](torch.cat((inputlist[-1], pflist[-1]), dim=1))
        x_1 = self.layers[1](x_1 + inputlist[1])
        x_1 = self.layers[2](x_1)

        x_1 = self.layers[3](torch.cat((x_1, pflist[1]), dim=1))
        x_1 = self.layers[4](x_1 + inputlist[0])
        x_1 = self.layers[5](x_1)
        x_1 = self.conv_out(torch.cat((x_1, pflist[0]), dim=1))

        return x_1


class FFNet(nn.Module):
    def __init__(self, rgb_inchannel, pf_channel, out_channel):
        super().__init__()
        self.encoder_rgb = Encoder(rgb_inchannel)
        self.encoder_point = Encoder(pf_channel + 1)
        self.decoder = Decoder(out_channel)

    def forward(self, rgb, pf):
        rgblist = self.encoder_rgb(rgb)
        pflist = self.encoder_point(pf)
        output = self.decoder(rgblist, pflist)
        return output


if __name__ == '__main__':
    model = FFNet(3, 16, 1)
    print("parameters {:.3f}M".format(
        sum(tensor.numel()
            for tensor in model.encoder_rgb.parameters()) / 1e6))
    print("parameters {:.3f}M".format(
        sum(tensor.numel()
            for tensor in model.encoder_point.parameters()) / 1e6))
    print("parameters {:.3f}M".format(
        sum(tensor.numel() for tensor in model.decoder.parameters()) / 1e6))
