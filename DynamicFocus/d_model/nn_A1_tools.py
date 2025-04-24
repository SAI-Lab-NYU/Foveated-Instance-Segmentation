import torch
import torchvision.transforms as transforms

def cd_cdf(tensor_x: torch.Tensor) -> torch.Tensor:
    """
    Cauchy Distribution CDF
    """
    return torch.arctan(tensor_x) / torch.pi + 0.5


def a_gd_cdf(tensor_x: torch.Tensor, a_gd_cdf_constant=torch.sqrt(torch.tensor(2. / torch.pi))) -> torch.Tensor:
    """
    Approximate Gaussian Distribution CDF
    """

    return torch.tanh(a_gd_cdf_constant * tensor_x) / 2. + 0.5


# def standardize_BxCxHxW(img_BxCxHxW: torch.Tensor):
#     B, C, H, W = img_BxCxHxW.shape
#     res_img_BxCxHxW = img_BxCxHxW
#     if H * W > 1:
#         value_mean = torch.mean(img_BxCxHxW, dim=[-2, -1], keepdim=True)
#         value_std = torch.std(img_BxCxHxW, dim=[-2, -1], keepdim=True)
#         res_img_BxCxHxW = (img_BxCxHxW - value_mean) / value_std
#     return res_img_BxCxHxW

def standardize_BxCxHxW(img_BxCxHxW: torch.Tensor):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
    return normalize(img_BxCxHxW)


def scale01_BxCxHxW(img_BxCxHxW: torch.Tensor):
    B, C, H, W = img_BxCxHxW.shape
    res_img_BxCxHxW = img_BxCxHxW
    if H * W > 1:
        value_max = torch.amax(img_BxCxHxW, dim=[-2, -1], keepdim=True)
        value_min = torch.amin(img_BxCxHxW, dim=[-2, -1], keepdim=True)

        img_BxCxHxW[:] = 1.0 - (value_max - img_BxCxHxW) / (value_max - value_min)

    return res_img_BxCxHxW

if __name__ == '__main__':
    B, C, H, W = 4, 3, 128, 128

    # Generate a random tensor with size BxCxHxW
    img_BxCxHxW = torch.rand((B, C, H, W))
    x = standardize_BxCxHxW(img_BxCxHxW)
