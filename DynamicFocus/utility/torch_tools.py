import numpy as np
import torch
import torch.nn.functional as F


def get_padding(H, W, HC, WC):
    pad_left = (WC - W) // 2
    pad_right = WC - W - pad_left
    pad_top = (HC - H) // 2
    pad_bottom = HC - H - pad_top
    return pad_left, pad_right, pad_top, pad_bottom


def add_alpha(img, alpha=None):
    # Determine the shape of the input image tensor
    *prefix_dims, C, H, W = img.shape

    # Create a new tensor to hold the result with an additional alpha channel
    res_img = torch.zeros((*prefix_dims, C + 1, H, W), dtype=torch.float32, device=img.device)

    # Copy the original image into the result tensor (excluding the alpha channel)
    res_img[..., :-1, :, :] = img

    # Handle the alpha channel
    if alpha is None:
        # If no alpha is provided, set alpha to all ones
        res_img[..., -1, :, :] = torch.ones((*prefix_dims, H, W), dtype=torch.float32, device=img.device)
    else:
        # Ensure alpha matches the dimensions of the image
        assert alpha.shape[-2:] == (H, W), "Alpha mask must have shape HxW"
        # If alpha is provided, insert it into the last channel
        res_img[..., -1, :, :] = alpha

    return res_img


def kernel_swap(view_XxHxWxKxK: torch.Tensor):
    view_XxKxKxHxW = view_XxHxWxKxK.transpose(-4, -2).transpose(-3, -1)
    return view_XxKxKxHxW


def cross_unfold(view_XxHxW: torch.Tensor, size_step=2):
    view_XxHxWxKxK = view_XxHxW.unfold(-2, size_step, size_step).unfold(-2, size_step, size_step)

    return view_XxHxWxKxK


def cross_fold(view_XxHxWxKxK: torch.Tensor):
    view_XxHxKxWxK = view_XxHxWxKxK.transpose(-2, -3)
    view_XxHxW = view_XxHxKxWxK.flatten(-4, -3).flatten(-2, -1)
    return view_XxHxW


def interpolate_int(x_BxHxW: torch.Tensor, degree=1):
    K = 2 ** degree
    x_BxHSxWSxKxK = cross_unfold(x_BxHxW, size_step=K)
    x_BxHSxWSxKK = x_BxHSxWSxKxK.flatten(start_dim=-2)
    max_value = x_BxHSxWSxKK.max().item() + 1

    x_BxHSxWS_np = np.argmax(np.apply_along_axis(lambda x: np.bincount(x, minlength=max_value), axis=-1, arr=x_BxHSxWSxKK.detach().cpu().numpy()), axis=-1)
    x_BxHSxWS = torch.from_numpy(x_BxHSxWS_np).to(dtype=x_BxHxW.dtype, device=x_BxHSxWSxKxK.device)
    return x_BxHSxWS


def gen_grid_mtx_2xHxW(H, W, device=None):
    idx_2xHxW = torch.zeros((2, H, W), dtype=torch.int64, device=device)
    idx_2xHxW[0, :, :] = torch.arange(H)[:, None].repeat(1, W)
    idx_2xHxW[1, :, :] = torch.arange(W)[None, :].repeat(H, 1)
    return idx_2xHxW


def gaussian_function(tensor, mean=0, std=1, device=None):
    # 计算高斯函数的系数，使用torch.sqrt进行计算
    seed = torch.ones(1, dtype=torch.float32, device=device)

    coefficient = 1 / (std * torch.sqrt(seed * 2 * torch.pi))
    # 计算指数部分，使用torch.exp进行计算
    exponent = torch.exp(-0.5 * ((tensor - mean) / std) ** 2)
    # 返回高斯函数的值
    return coefficient * exponent


def gen_focus_Gaussian_HxW(idx_H, idx_W, canvas_H, canvas_W, mean=0, std=512, device=None):
    idx_2xHxW = gen_grid_mtx_2xHxW(canvas_H, canvas_W, device=device)

    dist_HxW = torch.sqrt((idx_2xHxW[0, :, :] - idx_H) ** 2 + (idx_2xHxW[1, :, :] - idx_W) ** 2)

    valu_HxW = gaussian_function(dist_HxW, mean=mean, std=std, device=device)
    valu_HxW /= torch.max(valu_HxW)

    return valu_HxW


def str_tensor_shape(tensor: torch.Tensor):
    return 'x'.join(map(str, tensor.shape))


if __name__ == '__main__':
    pass
    x_BxHxW = torch.randint(0, 9, (8, 16), dtype=torch.int64)
    x_BxHxW = torch.randint(0, 2, (1, 8, 16), dtype=torch.bool)

    print(x_BxHxW.to(dtype=torch.int64))

    print(interpolate_int(x_BxHxW, degree=1).to(dtype=torch.int64))
