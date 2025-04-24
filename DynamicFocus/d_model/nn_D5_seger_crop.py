from typing import Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from d_model.nn_A0_utils import try_gpu, show_model_info
from d_model.nn_B1_croper import get_idxs_crop4
from d_model.nn_B4_Unet6 import UNet6
from utility.plot_tools import plt_imgshow, plt_show, plt_multi_imgshow
from utility.torch_tools import gen_grid_mtx_2xHxW
from utility.fctn import load_image


class SegerCrop(nn.Module):
    def __init__(self, base_module: Type[nn.Module], in_channels=4, out_channels=1, downsample_factor=8, cbase_seg=32, nlayer_seg=4):
        super(SegerCrop, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = out_channels

        self.downsample_factor = downsample_factor
        self.downsample_degree = int(np.log2(self.downsample_factor))

        self.alias = self.downsample_degree + 1

        self.gen_seg = base_module(in_channels=in_channels + 1, out_channels=out_channels, base_channels=cbase_seg, layer_num=nlayer_seg)

        self.ave_pool_d = nn.ModuleList()
        self.max_pool_d = nn.ModuleList()

        self.ave_pool_d.append(nn.Identity())
        self.max_pool_d.append(nn.Identity())

        for ds_degree in range(1, self.downsample_degree + 1):
            cur_downsample_factor = 2 ** (ds_degree)

            cur_ave_pool = nn.AvgPool2d(kernel_size=cur_downsample_factor, stride=cur_downsample_factor, padding=0)
            cur_max_pool = nn.MaxPool2d(kernel_size=cur_downsample_factor, stride=cur_downsample_factor, padding=0)

            self.ave_pool_d.append(cur_ave_pool)
            self.max_pool_d.append(cur_max_pool)

        self.transform_rgbaf = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5, 0.5], std=[0.229, 0.224, 0.225, 0.25, 0.25])  # 针对 ImageNet 进行标准化
        ])

    def get_idxs_crop_BxAx4(self, hidx_B: torch.Tensor, widx_B: torch.Tensor, H, W):
        assert hidx_B.shape == widx_B.shape
        B, = hidx_B.shape

        idxs_crop_BxAx4_s = []
        for b in range(B):
            idxs_crop_Ax4_s = []
            for a in range(self.alias):
                cur_ds_factor = 2 ** a
                pad = get_idxs_crop4(hidx_B[b].item(), widx_B[b].item(), H, W, H // cur_ds_factor, W // cur_ds_factor)
                idxs_crop_Ax4_s.append(pad)
            idxs_crop_BxAx4_s.append(idxs_crop_Ax4_s)

        idxs_crop_BxAx4 = torch.tensor(idxs_crop_BxAx4_s, dtype=torch.int64)
        return idxs_crop_BxAx4

    def gen_alias(self, any_BxCxHxW: torch.Tensor, idxs_crop_BxAx4: torch.Tensor, fctns: nn.ModuleList):
        B, C, H, W = any_BxCxHxW.shape
        target_device = any_BxCxHxW.device
        A = len(fctns)
        HS = H // self.downsample_factor
        WS = W // self.downsample_factor

        any_as_BxAxCxHSxWS = torch.zeros((B, A, C, HS, WS), dtype=torch.float32, device=target_device)

        for b in range(B):
            for a in range(self.alias):
                left, right, up, down = idxs_crop_BxAx4[b, a]

                any_as_BxAxCxHSxWS[b, a] = fctns[-(a + 1)](any_BxCxHxW[b, :, up:down, left:right])

        return any_as_BxAxCxHSxWS

    def gen_unalias(self, any_as_BxAxCxHSxWS: torch.Tensor, idxs_crop_BxAx4: torch.Tensor):

        # plt_multi_imgshow([x_ds_rgbaf_C1xHSxWS[:-1] for x_ds_rgbaf_C1xHSxWS in x_ds_rgbaf_BxAxC1xHSxWS[0]], row_col=(2, 2))
        # plt_show()

        B, A, C, HS, WS = any_as_BxAxCxHSxWS.shape
        target_device = any_as_BxAxCxHSxWS.device

        H = HS * self.downsample_factor
        W = WS * self.downsample_factor

        any_ua_BxCxHxW = torch.zeros((B, C, H, W), dtype=torch.float32, device=target_device)

        for a in range(A):
            cur_H = H // (2 ** a)
            cur_W = W // (2 ** a)

            any_as_curA_BxCxHcxWc = F.interpolate(any_as_BxAxCxHSxWS[:, a, :, :, :], (cur_H, cur_W), mode='bilinear', align_corners=True)

            for b in range(B):
                left, right, up, down = idxs_crop_BxAx4[b, a]

                any_ua_BxCxHxW[b, :, up:down, left:right] = any_as_curA_BxCxHcxWc[b]

            # plt_imgshow(any_ua_BxCxHxW[0, :-1, :, :])
            # plt_show()

        return any_ua_BxCxHxW

    def forward(self, x_BxCxHxW: torch.Tensor, x_Bx2: torch.Tensor):
        B, C, H, W = x_BxCxHxW.shape
        A = self.alias

        target_device = x_BxCxHxW.device

        # prepare focus map
        hidx_B = (x_Bx2[:, 0] * (H - 1))
        widx_B = (x_Bx2[:, 1] * (W - 1))

        grid_mtx_Bx2xHxW = gen_grid_mtx_2xHxW(H, W, device=target_device).unsqueeze(0).repeat(B, 1, 1, 1)
        dist_BxHxW = torch.sqrt((grid_mtx_Bx2xHxW[:, 0, :, :] - hidx_B[:, None, None]) ** 2 + (grid_mtx_Bx2xHxW[:, 1, :, :] - widx_B[:, None, None]) ** 2)

        dist_norm_Bx1xHxW = (dist_BxHxW / torch.amax(dist_BxHxW, dim=[-1, -2], keepdim=True)).unsqueeze(1)

        idxs_crop_BxAx4 = self.get_idxs_crop_BxAx4(hidx_B, widx_B, H, W).to(device=target_device)

        x_as_rgbaf_BxAxC1xHSxWS = torch.zeros(B, self.alias, C + 1, H // self.downsample_factor, W // self.downsample_factor).to(dtype=torch.float32, device=x_BxCxHxW.device)

        B, A, C1, HS, WS = x_as_rgbaf_BxAxC1xHSxWS.shape

        x_as_rgbaf_BxAxC1xHSxWS[:, :, :-1, :, :] = self.gen_alias(x_BxCxHxW, idxs_crop_BxAx4, self.ave_pool_d)
        x_as_rgbaf_BxAxC1xHSxWS[:, :, -1:, :, :] = self.gen_alias(dist_norm_Bx1xHxW, idxs_crop_BxAx4, self.max_pool_d)

        # gen seg map
        y_pred_as_BxAx1xHSxWS = torch.sigmoid(self.gen_seg(self.transform_rgbaf(x_as_rgbaf_BxAxC1xHSxWS.flatten(start_dim=0, end_dim=1)))).view(B, A, 1, HS, WS) * x_as_rgbaf_BxAxC1xHSxWS[:, :, -2:-1, :, :]

        return y_pred_as_BxAx1xHSxWS, idxs_crop_BxAx4, x_as_rgbaf_BxAxC1xHSxWS


if __name__ == '__main__':
    pass
    target_device = try_gpu()
    in_channels = 4  # rgba
    out_channels = 1
    canvas_H = 640
    canvas_W = 640
    sample_factor = 8
    B = 1

    # x_BxCxHxW = torch.randn(B, in_channels, canvas_H, canvas_W).to(device=target_device)
    # x_Bx2 = torch.randn(B, 2).to(device=target_device)

    fpath = r'D:\b_data_train\data_c_cook\lvis\train\sp1000\apple_c11_a616436_000000205724_393x331_0x0x107x107_4x426x640.uint8.S.png'
    x_BxCxHxW = load_image(fpath, 'RGBA').to(device=target_device).unsqueeze(0)

    x_Bx2 = torch.Tensor([[393 / canvas_H, 331 / canvas_W]]).to(device=target_device)

    e2e = SegerCrop(base_module=UNet6, in_channels=in_channels, out_channels=out_channels, downsample_factor=sample_factor, cbase_seg=16, nlayer_seg=4).to(target_device)

    y_pred_ds_BxAx1xHSxWS, idxs_crop_BxAx4, x_ds_rgbaf_BxAxC1xHSxWS = e2e(x_BxCxHxW, x_Bx2)

    print(x_BxCxHxW.shape)
    print(idxs_crop_BxAx4.shape)
    print(x_ds_rgbaf_BxAxC1xHSxWS.shape)
    print(y_pred_ds_BxAx1xHSxWS.shape)

    x_ua_rgbaf_BxC1xHxW = e2e.gen_unalias(x_ds_rgbaf_BxAxC1xHSxWS, idxs_crop_BxAx4)
