from d_model.nn_A0_utils import calc_model_memsize
from utility.torch_tools import str_tensor_shape

import torch
import torch.nn as nn


class ConvDown2D(nn.Module):
    """2D convolutional downsampling block: Convolution + BatchNorm + ReLU + Dropout"""

    def __init__(self, in_channels, out_channels, kernel=3, pad=1, step=1, dropout_rate=0.5):
        super(ConvDown2D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=step, padding=pad),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)  # Add Dropout after ReLU
        )

    def forward(self, x):
        return self.conv(x)


class ConvUp2D(nn.Module):
    """2D convolutional upsampling block: Transposed Convolution + BatchNorm + ReLU + Dropout"""

    def __init__(self, in_channels, out_channels, kernel=2, pad=0, step=2, dropout_rate=0.5):
        super(ConvUp2D, self).__init__()
        self.conv_up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel, stride=step, padding=pad),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)  # Add Dropout after ReLU
        )

    def forward(self, x):
        return self.conv_up(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation module"""

    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        y = self.global_avg_pool(x).view(batch_size, channels)
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y)).view(batch_size, channels, 1, 1)
        return x * y.expand_as(x)


class GlobalContextBlock(nn.Module):
    """Global Context Block for extracting global information"""

    def __init__(self, in_channels):
        super(GlobalContextBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 2, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        y = self.global_avg_pool(x).view(batch_size, channels)
        y = self.fc(y).view(batch_size, channels, 1, 1)
        return x * y.expand_as(x)


class UNet7(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=32, layer_num=4, dropout_rate=0.5):
        super(UNet7, self).__init__()
        self.layer_num = layer_num

        # Encoder
        self.enc_convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        encoder_channels = []

        current_in_channels = in_channels
        current_out_channels = base_channels

        for i in range(layer_num):
            self.enc_convs.append(ConvDown2D(current_in_channels, current_out_channels, dropout_rate=dropout_rate))
            encoder_channels.append(current_out_channels)
            self.pools.append(nn.MaxPool2d(2))
            current_in_channels = current_out_channels
            current_out_channels *= 2

        # Bottleneck with SEBlock and GlobalContextBlock
        self.bottleneck_conv = ConvDown2D(current_in_channels, current_out_channels, dropout_rate=dropout_rate)
        self.se_block = SEBlock(current_out_channels)
        self.global_context_block = GlobalContextBlock(current_out_channels)

        # Decoder
        self.up_convs = nn.ModuleList()
        self.dec_convs = nn.ModuleList()

        decoder_in_channels = current_out_channels
        for i in range(layer_num):
            decoder_out_channels = decoder_in_channels // 2

            self.up_convs.append(ConvUp2D(decoder_in_channels, decoder_out_channels, dropout_rate=dropout_rate))

            # The in_channels to dec_conv is decoder_out_channels + encoder_channels[layer_num - i -1]
            enc_channels = encoder_channels[layer_num - i - 1]
            dec_conv_in_channels = decoder_out_channels + enc_channels

            self.dec_convs.append(ConvDown2D(dec_conv_in_channels, decoder_out_channels, dropout_rate=dropout_rate))

            decoder_in_channels = decoder_out_channels

        self.conv2d_out = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc_features = []
        for i in range(self.layer_num):
            x = self.enc_convs[i](x)
            enc_features.append(x)
            x = self.pools[i](x)

        # Bottleneck
        x = self.bottleneck_conv(x)
        x = self.se_block(x)
        x = self.global_context_block(x)

        # Decoder
        for i in range(self.layer_num):
            x = self.up_convs[i](x)
            # Get the corresponding encoder feature map
            enc_feature = enc_features[self.layer_num - i - 1]
            x = torch.cat([x, enc_feature], dim=1)
            x = self.dec_convs[i](x)

        x = self.conv2d_out(x)
        return x


if __name__ == '__main__':
    # Example usage

    in_channels = 4
    out_channels = 1

    for Cbase in [8, 16, 32, 64]:
        for Nlayer in [3, 4, 5]:
            model = UNet7(in_channels=in_channels, out_channels=out_channels, base_channels=Cbase, layer_num=Nlayer)  # Example for RGB input and 2-class output

            # input_tensor = torch.randn((10, in_channels, 256, 256))  # Example input tensor (batch_size, channels, height, width)
            # output = model(input_tensor) {str_tensor_shape(output)}
            print(f"Cbase {Cbase} Nlayer {Nlayer} {calc_model_memsize(model, show=False):.4f}MB ")
