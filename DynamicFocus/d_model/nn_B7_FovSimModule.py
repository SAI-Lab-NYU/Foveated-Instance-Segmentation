import torch
from torch import nn

from d_model.nn_A0_utils import calc_model_memsize


class FovSimModule(nn.Module):
    def __init__(self, in_channels=3, out_channels=3,**kwargs):
        # in_channels: num of channels corresponds to input image channels, e.g. 3
        # out_channels: num of channels corresponds to num of sclaes tested
        super(FovSimModule, self).__init__()
        BN_MOMENTUM = 0.1
        self.fov_expand_1 = nn.Conv2d(in_channels=in_channels, out_channels=8 * out_channels, kernel_size=3, padding=1, bias=False)
        self.fov_expand_2 = nn.Conv2d(in_channels=8 * out_channels, out_channels=8 * out_channels, kernel_size=3, padding=1, bias=False)
        self.fov_squeeze_1 = nn.Conv2d(in_channels=8 * out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False)
        # bn
        self.norm1 = nn.BatchNorm2d(8 * out_channels, momentum=BN_MOMENTUM)
        self.norm2 = nn.BatchNorm2d(8 * out_channels, momentum=BN_MOMENTUM)
        self.norm3 = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.act = nn.ReLU6(inplace=False)

    def forward(self, x, reset_grad=True, train_mode=True):
        layer1 = self.act(self.norm1(self.fov_expand_1(x)))
        layer2 = self.act(self.norm2(self.fov_expand_2(layer1)))
        layer3 = self.norm3(self.fov_squeeze_1(layer2))
        output = layer3
        return output


if __name__ == '__main__':
    pass
    in_channels = 4
    out_channels = 1  # 请将 K 替换为您的类别数

    H = 160
    W = 160

    B = 5

    fsm = FovSimModule(in_channels, out_channels)
    calc_model_memsize(fsm)
    x_BxCxHxW = torch.randn(B, in_channels, H, W)

    y = fsm(x_BxCxHxW)
    print(y.shape)
