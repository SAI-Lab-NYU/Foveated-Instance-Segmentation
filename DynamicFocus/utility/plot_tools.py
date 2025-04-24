import os
import platform

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
from fontTools.unicodedata import block

system = platform.system()
if system == "Windows":
    matplotlib.use('TkAgg')  # 有图形界面，使用 TkAgg

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['axes.unicode_minus'] = False
plt.style.use('Solarize_Light2')

from matplotlib.colors import LinearSegmentedColormap

# Define the colors red, grey, and green
cmap_colors_rgg = ['green', 'grey', 'red']

# Create a colormap object
cmap_rgg = LinearSegmentedColormap.from_list("red_grey_green", cmap_colors_rgg)


def plt_imgshow(img_XxHxW: torch.Tensor, title='', ax=None):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    cax = None
    img_HxWxX = img_XxHxW
    if len(img_XxHxW.shape) == 3:
        img_HxWxX = img_XxHxW.permute(1, 2, 0)
    if isinstance(img_HxWxX, torch.Tensor):
        # print(img_XxHxW.shape)
        img_HxWxX_np = img_HxWxX.detach().cpu().numpy()
        cax = ax.imshow(img_HxWxX_np)
    else:
        # print(img_XxHxW.shape)
        cax = ax.imshow(img_HxWxX)

    cond1 = len(img_HxWxX.shape) == 3 and img_HxWxX.shape[2] == 1
    cond2 = len(img_HxWxX.shape) == 2
    if cond1 or cond2:
        if fig and cax:
            fig.colorbar(cax, ax=ax)

    ax.grid(False)
    return cax

def plt_multi_imgshow(imgs: list, titles: list = None, row_col: tuple = (1, 1)):
    """
    使用 plt_imgshow 显示多张图片。

    参数:
    - imgs: 图片的列表（作为 torch.Tensor 或 numpy 数组）
    - grid_shape: 一个 tuple (rows, cols) 定义网格布局
    - titles: 可选的图片标题列表
    """
    rows, cols = row_col
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 8, rows * 6))

    # 如果是多个图像，需要将 axes 展平
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for ax in axes.flatten():
        ax.grid(False)
        ax.set_visible(False)
    if titles is None:
        titles = [str(i) for i in range(len(imgs))]
    for i, (img, title) in enumerate(zip(imgs, titles)):
        if not img is None:
            axes[i].set_visible(True)
            cax = plt_imgshow(img, title, axes[i])  # 调用你已有的函数显示单张图像
            cond1 = len(img.shape) == 3 and img.shape[0] == 1
            cond2 = len(img.shape) == 2
            if cond1 or cond2:
                if fig and cax:
                    fig.colorbar(cax, ax=axes[i])

    plt.tight_layout()


plt_show = lambda: plt.show(block=True)
