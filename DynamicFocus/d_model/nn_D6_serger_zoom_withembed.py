'''
This file utilize embedding for focus point
'''

from pprint import pprint
from typing import Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torchvision import transforms

from d_model.nn_A0_utils import RAM
from d_model.nn_B0_deformed_sampler import get_grid_Bx2xHSxWS, deformed_unsampler, int_rount_scale_grid, gaussian_kernel
from d_model.nn_B1_croper import get_idxs_crop4
from utility.plot_tools import plt_show, plt_imgshow, plt_multi_imgshow

from utility.torch_tools import gen_grid_mtx_2xHxW
from d_model.nn_B4_Unet6 import UNet6
from d_model.nn_B8_embedding import FocusEmbedding

class SegerZoomEmbed(nn.Module):

    def __init__(self,
                 base_module_deformation: Type[nn.Module],
                 base_module: Type[nn.Module],
                 classify_module: Type[nn.Module],
                 in_channels=4,
                 out_channels=1,
                 class_num=2,
                 downsample_factor=4,
                 downsample_factor_deformation=8,
                 kernel_gridsp=40 + 1,
                 kernel_gblur=40 + 1,
                 kernel_priori=20 + 1,
                 cbase_seg0=4,
                 cbase_seg=64,
                 nlayer_seg0=2,
                 nlayer_seg=4,
                 priori=True,
                 square_focus=True):
        super(SegerZoomEmbed, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = out_channels

        self.downsample_factor = downsample_factor
        self.downsample_factor_deformation = downsample_factor_deformation

        self.kernel_gridsp = kernel_gridsp

        self.pad_gridsp = self.kernel_gridsp // 2

        self.gen_seg_0 = base_module_deformation(in_channels=in_channels + 1, out_channels=out_channels, base_channels=cbase_seg0, layer_num=nlayer_seg0)
        self.gen_seg = base_module(in_channels=in_channels + 1, out_channels=out_channels, base_channels=cbase_seg, layer_num=nlayer_seg)
        self.gen_cls = classify_module(in_channels=in_channels - 1 + 1, out_channels=class_num)

        self.ave_pool = nn.AvgPool2d(kernel_size=self.downsample_factor, stride=self.downsample_factor, padding=0)
        self.ave_pool_deformation = nn.AvgPool2d(kernel_size=self.downsample_factor_deformation, stride=self.downsample_factor_deformation, padding=0)

        sigma = kernel_gblur // 2
        self.gaussian_blur = transforms.GaussianBlur(kernel_size=kernel_gblur, sigma=(sigma, sigma))

        self.max_pool = nn.MaxPool2d(kernel_size=self.downsample_factor, stride=self.downsample_factor, padding=0)
        self.focus_embed = FocusEmbedding(1, 160, 160)

        self.priori = priori
        self.kernel_priori = kernel_priori
        self.square_focus = square_focus

        self.transform_rgbaf = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5, 0.5], std=[0.229, 0.224, 0.225, 0.25, 0.25])  # 针对 ImageNet 进行标准化
        ])

        self.init_value = 3.0
        priori_gaussian_1xKxK = gaussian_kernel(size=self.kernel_priori, sigma=self.kernel_priori // 6)[None, :, :]
        p_max = torch.amax(priori_gaussian_1xKxK, dim=[-1, -2], keepdim=True)
        self.priori_gaussian_1xKxK = priori_gaussian_1xKxK / p_max

    def forward(self, x_BxCxHxW: torch.Tensor, x_Bx2: torch.Tensor):
        B, C, H, W = x_BxCxHxW.shape

        # H//4, W//4
        HS = H // self.downsample_factor  # 64
        WS = W // self.downsample_factor  # 128

        target_device = x_BxCxHxW.device

        # initialize rgbaf 5 channegl
        x_ds_rgbaf_BxC1xHSxWS = torch.zeros(B, C + 1, HS, WS).to(dtype=torch.float32, device=x_BxCxHxW.device)

        print(x_Bx2.shape)
        focusmap_Bx1xHxW = self.focus_embed(x_Bx2)
        print(focusmap_Bx1xHxW.shape)

        # assign RGB color map
        x_ds_rgbaf_BxC1xHSxWS[:, :-1, :, :] = self.downsample_x_real_BxCxHSxWS(x_BxCxHxW)
        # assign focus map
        # min_x = torch.amin(dist_BxHxW, dim=[-1, -2], keepdim=True)
        # max_x = torch.amax(dist_BxHxW, dim=[-1, -2], keepdim=True)

        x_ds_rgbaf_BxC1xHSxWS[:, -1, :, :] = focusmap_Bx1xHxW[:, 0, :, :]

        if self.square_focus:
            x_ds_rgbaf_BxC1xHSxWS[:, -1, :, :] = x_ds_rgbaf_BxC1xHSxWS[:, -1, :, :] ** 2

        # make priori
        priori_Bx1xHSxWS = None

        if self.priori:
            priori_Bx1xHSxWS = torch.zeros(B, 1, HS, WS).to(dtype=torch.float32, device=target_device)

            for bidx, (hidx_1, widx_1) in enumerate(zip(hidx_B.to(int).tolist(), widx_B.to(int).tolist())):
                # print(hidx_1, widx_1, end=', ')

                left, right, up, bottom = get_idxs_crop4(hidx_1, widx_1, HS, WS, self.kernel_priori, self.kernel_priori)
                priori_Bx1xHSxWS[bidx, :, up:bottom, left:right] = self.priori_gaussian_1xKxK

                priori_Bx1xHSxWS[bidx, :, hidx_1, widx_1] = 0.0

                # plt_imgshow(priori_Bx1xHSxWS[bidx, :, :, :])
                # plt_show()

            # plt_imgshow(priori_Bx1xHSxWS[0, :, :, :])
            # plt_show()
            priori_Bx1xHSxWS = (2.0 * priori_Bx1xHSxWS - 1) * self.init_value

            # plt_imgshow(priori_Bx1xHSxWS[0, :, :, :])
            # plt_show()

        # plt_imgshow(priori_Bx1xHSxWS[0, :, :, :])
        # plt_show()

        # gen density map
        out0_Bx1xHSxWS = self.gen_seg_0(self.transform_rgbaf(x_ds_rgbaf_BxC1xHSxWS))
        if priori_Bx1xHSxWS is not None:
            # plt_multi_imgshow([out0_Bx1xHSxWS[0, :, :, :], priori_Bx1xHSxWS[0, :, :, :]], row_col=(1, 2))
            # plt_show()

            out0_Bx1xHSxWS = out0_Bx1xHSxWS + priori_Bx1xHSxWS

            # plt_imgshow(out0_Bx1xHSxWS[0, :, :, :])
            # plt_show()

        dmap_pred_ds_Bx1xHSxWS = torch.sigmoid(out0_Bx1xHSxWS) * x_ds_rgbaf_BxC1xHSxWS[:, -2:-1, :, :]

        # gen grid
        dm_v_Bx1xHSPxWSP = F.pad(self.ave_pool_deformation(self.gaussian_blur(dmap_pred_ds_Bx1xHSxWS)), (self.pad_gridsp, self.pad_gridsp, self.pad_gridsp, self.pad_gridsp), mode='replicate')
        grid_pred_Bx2xHSxWS = get_grid_Bx2xHSxWS(dm_v_Bx1xHSPxWSP, HS // self.downsample_factor_deformation, WS // self.downsample_factor_deformation, kernel_size=self.kernel_gridsp)

        grid_pred_BxHSxWSx2 = torch.flip(F.interpolate(grid_pred_Bx2xHSxWS, (HS, WS), mode='bilinear', align_corners=True).permute(0, 2, 3, 1), dims=[-1])

        x_gs_rgbaf_BxC1xHSxWS = torch.zeros(B, C + 1, HS, WS).to(dtype=torch.float32, device=x_BxCxHxW.device)
        # assign RGB color map
        x_gs_rgbaf_BxC1xHSxWS[:, :-1, :, :] = F.grid_sample(x_BxCxHxW, grid_pred_BxHSxWSx2, mode='bilinear', align_corners=True)
        # assign focus map

        x_gs_rgbaf_BxC1xHSxWS[:, -1, :, :] = F.grid_sample(focusmap_Bx1xHxW, grid_pred_BxHSxWSx2, mode='bilinear', align_corners=True)[:, 0, :, :]

        out1_Bx1xHSxWS = self.gen_seg(self.transform_rgbaf(x_gs_rgbaf_BxC1xHSxWS))

        y_pred_gs_Bx1xHSxWS = torch.sigmoid(out1_Bx1xHSxWS) * x_gs_rgbaf_BxC1xHSxWS[:, -2:-1, :, :]

        y_pred_BxK = torch.softmax(self.gen_cls(x_gs_rgbaf_BxC1xHSxWS[:,[0, 1, 2, 4]] * x_gs_rgbaf_BxC1xHSxWS[:, -2:-1, :, :]), dim=1)

        # y_pred_BxK = None

        # plt_multi_imgshow([x_gs_rgbaf_BxC1xHSxWS[i, :-1] for i in range(B)], row_col=(1, B))
        # plt_show()
        # plt_multi_imgshow([x_gs_rgbaf_BxC1xHSxWS[i, -1:] for i in range(B)], row_col=(1, B))
        # plt_show()

        del priori_Bx1xHSxWS
        RAM().gc()

        return y_pred_gs_Bx1xHSxWS, grid_pred_BxHSxWSx2, dmap_pred_ds_Bx1xHSxWS, x_ds_rgbaf_BxC1xHSxWS, x_gs_rgbaf_BxC1xHSxWS, y_pred_BxK

    def downsample_x_real_BxCxHSxWS(self, x_BxCxHxW: torch.Tensor):
        B, C, H, W = x_BxCxHxW.shape
        HS = H // self.downsample_factor  # 64
        WS = W // self.downsample_factor  # 128
        return F.interpolate(x_BxCxHxW, (HS, WS), mode='bilinear', align_corners=True)

    def downsample_y_maxpool_real_Bx1xHSxWS(self, y_real_Bx1xHxW: torch.Tensor):
        return self.max_pool(y_real_Bx1xHxW)

    def downsample_y_gridsp_real_Bx1xHSxWS(self, y_real_Bx1xHxW: torch.Tensor, grid_pred_BxHSxWSx2: torch.Tensor):
        return F.grid_sample(y_real_Bx1xHxW, grid_pred_BxHSxWSx2, mode='bilinear', align_corners=True)

    def output_y_pred_Bx1xHxW(self, y_pred_Bx1xHSxWS: torch.Tensor, grid_pred_BxHSxWSx2: torch.Tensor, x_BxCxHxW: torch.Tensor):
        B, _, HS, WS = y_pred_Bx1xHSxWS.shape
        H = HS * self.downsample_factor
        W = WS * self.downsample_factor

        grid_pred_BxHSxWSx2 = torch.flip(grid_pred_BxHSxWSx2, dims=[-1])
        grid_pred_Bx2xHSxWS = grid_pred_BxHSxWSx2.permute(0, 3, 1, 2)
        grid_pred_Bx2xHSxWS = int_rount_scale_grid(grid_pred_Bx2xHSxWS, H, W)

        return deformed_unsampler(y_pred_Bx1xHSxWS, grid_pred_Bx2xHSxWS, H, W) * x_BxCxHxW[:, -1:, :, :]


if __name__ == '__main__':
    pass
    in_channels = 3
    out_channels = 41
    canvas_H = 256
    canvas_W = 512
    sample_factor = 4
    kernel_size = 64 + 1
    B = 6
    injectD = 2

    x_BxCxHxW = torch.randn(B, in_channels, canvas_H, canvas_W)
    x_Bx2 = torch.randn(B, 2)
    e2e = SegerZoom(in_channels=in_channels, out_channels=out_channels, H=canvas_H, W=canvas_W, downsample_factor=sample_factor)

    ys_BxKxHSxWS, grid_BxHSxWSx2 = e2e(x_BxCxHxW, x_Bx2)

    # plt_multi_imgshow(imgs=[label_x_BxKxHxW[0, :, :, 0], label_x_BxKxHxW[0, :, :, 1]], titles=['w', 'h'], row_col=(2, 1))
    # plt.show(block=True)
