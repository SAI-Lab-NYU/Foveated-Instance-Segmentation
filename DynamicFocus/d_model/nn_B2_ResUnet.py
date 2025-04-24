from xmlrpc.client import Fault

import torch
import torch.nn as nn
import torch.nn.functional as F

from d_model.nn_A0_utils import calc_model_memsize
from utility.torch_tools import str_tensor_shape


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class ResUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64, layer_num=4):
        super(ResUNet, self).__init__()
        self.base_channels = base_channels
        self.layer_num = layer_num

        # Encoding path
        self.encoders = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        in_channels = in_channels  # Start with the specified input channels
        for i in range(layer_num):
            out_channels = base_channels * (2 ** i)
            self.encoders.append(ResidualBlock(in_channels, out_channels))
            in_channels = out_channels

        # Bottleneck
        self.bottleneck = ResidualBlock(in_channels, in_channels * 2)

        # Decoding path
        self.decoders = nn.ModuleList()
        for i in range(layer_num):
            in_channels = in_channels * 2 if i == 0 else out_channels
            out_channels = in_channels // 2
            self.decoders.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2))
            self.decoders.append(ResidualBlock(in_channels, out_channels))

        # Final output
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)  # Output channels defined by parameter

    def forward(self, x):
        # Encoder path with skip connections
        skip_connections = []
        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        for i in range(self.layer_num):
            x = self.decoders[2 * i](x)  # Upsampling
            skip_connection = skip_connections[-(i + 1)]
            x = torch.cat((x, skip_connection), dim=1)  # Concatenate skip connection
            x = self.decoders[2 * i + 1](x)  # Residual block

        # Final output layer
        x = self.final_conv(x)
        return x


if __name__ == '__main__':
    # Example usage

    in_channels = 4
    out_channels = 1

    for Cbase in [4, 8, 16, 32, 64, 128]:
        for Nlayer in [2, 3, 4, 5]:
            model = ResUNet(in_channels=in_channels, out_channels=out_channels, base_channels=Cbase, layer_num=Nlayer)  # Example for RGB input and 2-class output

            input_tensor = torch.randn((10, in_channels, 256, 256))  # Example input tensor (batch_size, channels, height, width)
            output = model(input_tensor)
            print(f"Cbase {Cbase} Nlayer {Nlayer} {calc_model_memsize(model, show=False):.4f} {str_tensor_shape(output)}")
