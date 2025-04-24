import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from d_model.nn_A0_utils import RAM
from utility.fctn import load_image
from utility.plot_tools import *

from utility.torch_tools import gen_grid_mtx_2xHxW


def gaussian_kernel(size=5, sigma=1.0, device=None):
    """Generates a 2D Gaussian kernel using PyTorch, with the ability to specify device."""
    # Create a 2D grid of (x, y) coordinates
    coords = torch.linspace(-(size // 2), size // 2, size, device=device)
    x, y = torch.meshgrid(coords, coords, indexing='ij')  # Add 'indexing' argument

    # Compute the Gaussian function
    kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    # Normalize the kernel so that the sum of all elements is 1
    kernel /= kernel.sum()

    return kernel


def get_grid_Bx2xHSxWS(dm_v_Bx1xHSPxWSP: torch.Tensor, canvas_sample_H, canvas_sample_W, kernel_size=64 + 1):
    mgpu = RAM()

    target_device = dm_v_Bx1xHSPxWSP.device

    pad = kernel_size // 2

    # unfold operation, apply kernelsize on the whole tensor
    mgpu.dm_va_Bx1xHSxWSxKxK = dm_v_Bx1xHSPxWSP.unfold(-2, kernel_size, 1).unfold(-2, kernel_size, 1)

    # initialize gaussian kernel
    mgpu.kernel_w_Bx1x1x1xKxK = gaussian_kernel(size=kernel_size, sigma=kernel_size // 2, device=target_device)[None, None, None, None, :, :]

    # convolution
    mgpu.dm_conv_v_Bx1xHSxWSxKxK = mgpu.dm_va_Bx1xHSxWSxKxK * mgpu.kernel_w_Bx1x1x1xKxK

    # normalization
    mgpu.dm_conv_v_Bx1xHSxWSxKxK = (mgpu.dm_conv_v_Bx1xHSxWSxKxK + 1e-6) / torch.sum(mgpu.dm_conv_v_Bx1xHSxWSxKxK + 1e-6, dim=[-2, -1], keepdim=True)

    # idx_2xHSxWS = gen_idx_mtx_2xHxW(canvas_sample_H, canvas_sample_W, device=target_device).to(dtype=torch.float32)
    # idx_2xHSPxWSP = F.pad(idx_2xHSxWS, (pad, pad, pad, pad), mode='replicate')

    # inilialize grid matrix
    mgpu.grid_2xHSPxWSP = gen_grid_mtx_2xHxW(canvas_sample_H + 2 * pad, canvas_sample_W + 2 * pad, device=dm_v_Bx1xHSPxWSP.device).to(dtype=torch.float32)
    mgpu.grid_2xHSPxWSP[:, :, :] -= pad

    mgpu.grid_2xHSxWSxKxK = mgpu.grid_2xHSPxWSP.unfold(-2, kernel_size, 1).unfold(-2, kernel_size, 1)
    mgpu.grid_1x2xHSxWSxKxK = mgpu.grid_2xHSxWSxKxK[None, :, :, :, :, :]

    mgpu.grid_Bx2xHSxWS = torch.einsum('bphwkj,qthwkj->bthw', mgpu.dm_conv_v_Bx1xHSxWSxKxK, mgpu.grid_1x2xHSxWSxKxK)

    # normalize to [-1,1]
    mgpu.grid_Bx2xHSxWS[:, 0, :, :] /= canvas_sample_H - 1
    mgpu.grid_Bx2xHSxWS[:, 1, :, :] /= canvas_sample_W - 1
    mgpu.grid_Bx2xHSxWS = 2.0 * mgpu.grid_Bx2xHSxWS - 1.0

    # grid_BxHSxWSx2 = mgpu.grid_Bx2xHSxWS.permute(0, 2, 3, 1)
    # grid_BxHSxWSx2 = torch.flip(grid_BxHSxWSx2, dims=[-1])

    del mgpu.dm_va_Bx1xHSxWSxKxK
    del mgpu.kernel_w_Bx1x1x1xKxK
    del mgpu.dm_conv_v_Bx1xHSxWSxKxK
    del mgpu.grid_2xHSPxWSP
    del mgpu.grid_2xHSxWSxKxK
    del mgpu.grid_1x2xHSxWSxKxK
    # del mgpu.grid_Bx2xHSxWS

    mgpu.gc()

    return mgpu.grid_Bx2xHSxWS


def int_rount_scale_grid(grid_Bx2xHSxWS: torch.Tensor, canvas_H, canvas_W):
    """
    HS : canvas_sampler_H
    WS : canvas_sampler_W

    grid_Bx2xHSxWS[:, 0, :, :] in [0, canvas_H - 1]
    grid_Bx2xHSxWS[:, 1, :, :] in [0, canvas_W - 1]
    """
    # [-1,1] to [0,1]
    grid_Bx2xHSxWS = 0.5 * (grid_Bx2xHSxWS + 1.0)
    grid_Bx2xHSxWS[:, 0, :, :] *= canvas_H - 1
    grid_Bx2xHSxWS[:, 1, :, :] *= canvas_W - 1

    grid_Bx2xHSxWS[:, 0, :, :] = torch.clip(grid_Bx2xHSxWS[:, 0, :, :], 0, canvas_H - 1)
    grid_Bx2xHSxWS[:, 1, :, :] = torch.clip(grid_Bx2xHSxWS[:, 1, :, :], 0, canvas_W - 1)

    grid_Bx2xHSxWS = grid_Bx2xHSxWS.to(dtype=torch.int64)

    return grid_Bx2xHSxWS

# def deformed_unsampler(label_sample_BxKxHSxWS, grid_Bx2xHSxWS, canvas_H, canvas_W):
#     grid_Bx2xHSxWS = grid_Bx2xHSxWS.to(dtype = torch.float32)
#     grid_Bx2xHSxWS = F.interpolate(grid_Bx2xHSxWS, size=(canvas_H, canvas_W), mode='bilinear', align_corners=True)

#     grid_Bx2xHSxWS = grid_Bx2xHSxWS.permute(0, 2, 3, 1)

#     label_sample_BxKxHxW = F.grid_sample(label_sample_BxKxHSxWS, grid_Bx2xHSxWS, mode='nearest', align_corners=True)

#     return label_sample_BxKxHxW


'''
Manually grid upsampler
'''
def deformed_unsampler(label_sample_BxKxHSxWS: torch.Tensor, grid_Bx2xHSxWS: torch.Tensor, canvas_H, canvas_W):

    target_device = grid_Bx2xHSxWS.device

    B, K, HS, WS = label_sample_BxKxHSxWS.shape

    label_sample_BxKxHSxWS = label_sample_BxKxHSxWS.to(device=target_device)
    grid_Bx2xHSxWS = grid_Bx2xHSxWS.to(device=target_device)

    label_xa_BxK1xHxW = torch.zeros((B, K + 1, canvas_H, canvas_W), dtype=torch.float32, device=target_device)

    # Get the coordinates directly without flattening
    B_coords_BxHSxWS = torch.arange(B, device=target_device).view(B, 1, 1).expand(B, HS, WS)
    h_coords_BxHSxWS = grid_Bx2xHSxWS[:, 0, :, :]  # shape: [B, HS, WS]
    w_coords_BxHSxWS = grid_Bx2xHSxWS[:, 1, :, :]  # shape: [B, HS, WS]

    label_xa_BxK1xHxW[B_coords_BxHSxWS, -1, h_coords_BxHSxWS, w_coords_BxHSxWS] = 1.0

    B_coords_BxCxHSxWS = B_coords_BxHSxWS.view(B, 1, HS, WS).expand(B, K, HS, WS)
    h_coords_BxCxHSxWS = h_coords_BxHSxWS.view(B, 1, HS, WS).expand(B, K, HS, WS)
    w_coords_BxCxHSxWS = w_coords_BxHSxWS.view(B, 1, HS, WS).expand(B, K, HS, WS)
    C_coords_BxCxHSxWS = torch.arange(K, device=target_device).view(1, K, 1, 1).expand(B, K, HS, WS)

    label_xa_BxK1xHxW[B_coords_BxCxHSxWS, C_coords_BxCxHSxWS, h_coords_BxCxHSxWS, w_coords_BxCxHSxWS] = label_sample_BxKxHSxWS

    label_xa_BxK1xHxW = label_xa_BxK1xHxW.detach().cpu().numpy()

    for bid in range(B):
        # print(bid)

        mask_zeros_HxW = label_xa_BxK1xHxW[bid, -1, :, :] == 0.0

        distances, nearest_indices = distance_transform_edt(mask_zeros_HxW, return_indices=True)

        for cid in range(K):
            label_channel = label_xa_BxK1xHxW[bid, cid, :, :]
            label_channel[mask_zeros_HxW] = label_channel[nearest_indices[0][mask_zeros_HxW], nearest_indices[1][mask_zeros_HxW]]

    return torch.from_numpy(label_xa_BxK1xHxW[:, :-1, :, :]).to(device=target_device)


if __name__ == '__main__':
    # 示例使用
    canvas_H = 128
    canvas_W = 256

    canvas_sample_H = 64
    canvas_sample_W = 128

    kernel_size = 64 + 1

    image_rgb_Bx3xHxW = load_image(r'/home/lwx/b_data_train/data_a_raw/leftImg8bit_trainvaltest/leftImg8bit/train/darmstadt/darmstadt_000000_000019_leftImg8bit.png')[None, :, :, :]

    padding = kernel_size // 2

    HSP = canvas_sample_H + 2 * padding
    WSP = canvas_sample_W + 2 * padding
    HSPD = HSP // 2
    WSPD = WSP // 2
    image_rgb_Bx3xHDxWD = F.interpolate(image_rgb_Bx3xHxW, size=(HSPD, WSPD), mode='bilinear', align_corners=True)

    # dm_v_Bx1xHSPDxWSPD = gen_focus_Gaussian_HxW(30, 44, HSPD, WSPD, mean=0, std=4, device=MM_device_gpu)[None, None, :, :]
    # # dm_v_Bx1xHSPDxWSPD = torch.ones((HSPD, WSPD), device=MM_device_gpu)[None, None, :, :]
    #
    # dm_v_Bx1xHSPxWSP = F.interpolate(dm_v_Bx1xHSPDxWSPD, size=(HSP, WSP), mode='bilinear', align_corners=True)
    # image_sample_Bx3xHSxWS, idx_ds_v_Bx2xHSxWS = get_grid_BxHSxWSx2(dm_v_Bx1xHSPxWSP, image_rgb_Bx3xHxW, canvas_sample_H=canvas_sample_H, canvas_sample_W=canvas_sample_W, kernel_size=kernel_size)
    #
    # irs_idx_ds_v_Bx2xHSxWS = int_rount_scale_idx(idx_ds_v_Bx2xHSxWS, canvas_H, canvas_W)
    #
    # label_sample_BxCxHSxWS = image_sample_Bx3xHSxWS
    #
    # deformed_unsampler(irs_idx_ds_v_Bx2xHSxWS, label_sample_BxCxHSxWS, canvas_H, canvas_W)
    #
    # plt_imgshow(image_sample_Bx3xHSxWS[0, :, :, :], title=f'Deformed Sampler Output (Down-sampled Image) {list(image_sample_Bx3xHSxWS[0, :, :, :].shape)}')
    # plt_imgshow(dm_v_Bx1xHSPxWSP[0, :, :, :], title=f'Deformed Sampler Input (Density Map) {list(dm_v_Bx1xHSPxWSP[0, :, :, :].shape)}')
    # plt_imgshow(dm_v_Bx1xHSPDxWSPD[0, :, :, :], title=f'Deformation Module Output {list(dm_v_Bx1xHSPDxWSPD[0, :, :, :].shape)}')
    # plt_imgshow(image_rgb_Bx3xHDxWD[0, :, :, :], title=f'Deformation Module Input {list(image_rgb_Bx3xHDxWD[0, :, :, :].shape)}')
    # plt_imgshow(image_rgb_Bx3xHxW[0, :, :, :], title=f'Original Input {list(image_rgb_Bx3xHxW[0, :, :, :].shape)}')
    # plt.show(block=True)

    #
    #
    # plt_imgshow(dm_v_Bx1xHMxWM[0, 0, :, :], title=f'density map {dm_v_Bx1xHMxWM[0, 0, :, :].shape}')
    # # plt.show(block=True)
    #
    #
    #
    #
    # image_rgb_Bx4xHxW = add_alpha(image_rgb_Bx3xHxW)
    #
    # # image_rgb_Bx4xHxW[0, 0, torch.flatten(idx_ds_v_Bx2xHSxWS[0, 0, :, :]), torch.flatten(idx_ds_v_Bx2xHSxWS[0, 1, :, :])] = 1.0
    # # image_rgb_Bx4xHxW[0, 1, torch.flatten(idx_ds_v_Bx2xHSxWS[0, 0, :, :]), torch.flatten(idx_ds_v_Bx2xHSxWS[0, 1, :, :])] = 0.0
    # # image_rgb_Bx4xHxW[0, 2, torch.flatten(idx_ds_v_Bx2xHSxWS[0, 0, :, :]), torch.flatten(idx_ds_v_Bx2xHSxWS[0, 1, :, :])] = 1.0
    #
    # image_rgb_Bx4xHxW[:, -1, :, :] = 0.5
    # image_rgb_Bx4xHxW[0, -1, torch.flatten(idx_ds_v_Bx2xHSxWS[0, 0, :, :]), torch.flatten(idx_ds_v_Bx2xHSxWS[0, 1, :, :])] = 1.0
    #
    # plt_imgshow(image_rgb_Bx4xHxW[0, :, :, :], title='sample points')
    #
    # plt_imgshow(image_unsample_rgb_Bx4xHxW[0], title=f'unsampled map {image_unsample_rgb_Bx4xHxW[0].shape}')
    # # plt_imgshow(idx_ds_v_BxHSxWSx2[0, :, :, 0])
    # # plt_imgshow(idx_ds_v_BxHSxWSx2[0, :, :, 1])
    #
    # # plt.show(block=True)
    #
    # plt_imgshow(image_sample_Bx3xHSxWS[0], title=f'sampled map {image_sample_Bx3xHSxWS[0].shape}')
    # plt.show(block=True)
    # # 输出矩阵形状
    # print(image_rgb_Bx3xHxW.shape)  # 3xHxW
