from typing import Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from d_model.nn_A0_utils import try_gpu
from utility.torch_tools import gen_grid_mtx_2xHxW


class SegerAverage(nn.Module):
    def __init__(self, base_module: Type[nn.Module], in_channels=4, out_channels=1, downsample_factor=16, cbase_seg=32, nlayer_seg=4):
        super(SegerAverage, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = out_channels

        self.downsample_factor = downsample_factor

        self.gen_seg = base_module(in_channels=in_channels + 1, out_channels=out_channels, base_channels=cbase_seg, layer_num=nlayer_seg)

        self.ave_pool = nn.AvgPool2d(kernel_size=self.downsample_factor, stride=self.downsample_factor, padding=0)

        self.max_pool = nn.MaxPool2d(kernel_size=self.downsample_factor, stride=self.downsample_factor, padding=0)

        self.transform_rgbaf = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5, 0.5], std=[0.229, 0.224, 0.225, 0.25, 0.25])  # 针对 ImageNet 进行标准化
        ])

    def forward(self, *data, method='forward'):
        if method == 'forward':
            x_BxCxHxW, x_Bx2 = data
            B, C, H, W = x_BxCxHxW.shape

            HS = H // self.downsample_factor  # 64
            WS = W // self.downsample_factor  # 128

            target_device = x_BxCxHxW.device

            x_ds_rgbf_BxC1xHSxWS = torch.zeros(B, C + 1, HS, WS).to(dtype=torch.float32, device=x_BxCxHxW.device)

            # prepare focus map
            hidx_B = (x_Bx2[:, 0] * (HS - 1))
            widx_B = (x_Bx2[:, 1] * (WS - 1))
            grid_mtx_Bx2xHxW = gen_grid_mtx_2xHxW(HS, WS, device=target_device).unsqueeze(0).repeat(B, 1, 1, 1)
            dist_BxHxW = torch.sqrt((grid_mtx_Bx2xHxW[:, 0, :, :] - hidx_B[:, None, None]) ** 2 + (grid_mtx_Bx2xHxW[:, 1, :, :] - widx_B[:, None, None]) ** 2)

            # assign RGB color map
            x_ds_rgbf_BxC1xHSxWS[:, :-1, :, :] = self.downsample_x_real_BxCxHSxWS(x_BxCxHxW)
            # assign focus map
            x_ds_rgbf_BxC1xHSxWS[:, -1, :, :] = dist_BxHxW / torch.amax(dist_BxHxW, dim=[-1, -2], keepdim=True)

            # gen density map
            y_pred_ds_Bx1xHSxWS = torch.sigmoid(self.gen_seg(self.transform_rgbaf(x_ds_rgbf_BxC1xHSxWS))) * x_ds_rgbf_BxC1xHSxWS[:, -2:-1, :, :]

            return y_pred_ds_Bx1xHSxWS, x_ds_rgbf_BxC1xHSxWS, None

        elif method == 'downsample_x_real_BxCxHSxWS':
            x_BxCxHxW, = data
            return self.downsample_x_real_BxCxHSxWS(x_BxCxHxW)

        elif method == 'downsample_y_maxpool_real_Bx1xHSxWS':
            y_real_Bx1xHxW, = data
            return self.downsample_y_maxpool_real_Bx1xHSxWS(y_real_Bx1xHxW)

        elif method == 'output_y_pred_Bx1xHxW':
            y_pred_Bx1xHSxWS, = data
            return self.output_y_pred_Bx1xHxW(y_pred_Bx1xHSxWS)

    def downsample_x_real_BxCxHSxWS(self, x_BxCxHxW: torch.Tensor):
        return self.ave_pool(x_BxCxHxW)

    def downsample_y_maxpool_real_Bx1xHSxWS(self, y_real_Bx1xHxW: torch.Tensor):
        return self.max_pool(y_real_Bx1xHxW)

    def output_y_pred_Bx1xHxW(self, y_pred_Bx1xHSxWS: torch.Tensor):
        B, _, HS, WS = y_pred_Bx1xHSxWS.shape
        H = HS * self.downsample_factor
        W = WS * self.downsample_factor

        return F.interpolate(y_pred_Bx1xHSxWS, (H, W), mode='bilinear', align_corners=True)


if __name__ == '__main__':
    pass
    target_device = try_gpu()
    in_channels = 3
    out_channels = 41
    canvas_H = 256
    canvas_W = 512
    sample_factor = 4
    kernel_size = 64 + 1
    B = 6
    injectD = 2

    x_BxCxHxW = torch.randn(B, in_channels, canvas_H, canvas_W).to(device=target_device)
    x_Bx2 = torch.randn(B, 2).to(device=target_device)
    e2e = SegerAverage(in_channels=in_channels, out_channels=out_channels, H=canvas_H, W=canvas_W, downsample_factor=sample_factor).to(target_device)

    ys_BxKxHSxWS, grid_BxHSxWSx2 = e2e(x_BxCxHxW, x_Bx2)

    # plt_multi_imgshow(imgs=[label_x_BxKxHxW[0, :, :, 0], label_x_BxKxHxW[0, :, :, 1]], titles=['w', 'h'], row_col=(2, 1))
    # plt.show(block=True)
