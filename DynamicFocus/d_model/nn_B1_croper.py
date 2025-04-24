import random

import torch

from utility.plot_tools import plt_imgshow, plt_show


def get_idxs_crop4(idx_h: int, idx_w: int, H_canvas: int, W_canvas: int, H_crop: int, W_crop: int) -> tuple[int, int, int, int]:
    if H_crop >= H_canvas or W_crop >= W_canvas:
        return 0, W_canvas, 0, H_canvas

    # 确保 idx_h 和 idx_w 合法，并且 H_crop 和 W_crop 为偶数
    idx_h = min(max(0, idx_h), H_canvas - 1)
    idx_w = min(max(0, idx_w), W_canvas - 1)

    # 使用随机四舍五入来确定中心点的偏移
    idx_h = idx_h + int(round(random.random())) if H_crop % 2 == 0 else idx_h
    idx_w = idx_w + int(round(random.random())) if W_crop % 2 == 0 else idx_w

    # 计算上下和左右的偏移量
    half_H_crop = H_crop // 2
    half_W_crop = W_crop // 2

    # 假设没有超出边界的裁剪区域
    up = idx_h - half_H_crop
    bottom = idx_h + half_H_crop + H_crop % 2
    left = idx_w - half_W_crop
    right = idx_w + half_W_crop + W_crop % 2

    # 检查并调整边界，确保裁剪区域在画布范围内
    if up < 0: up, bottom = 0, H_crop
    if bottom > H_canvas: bottom, up = H_canvas, H_canvas - H_crop
    if left < 0: left, right = 0, W_crop
    if right > W_canvas: right, left = W_canvas, W_canvas - W_crop

    return left, right, up, bottom


if __name__ == '__main__':
    pass
    idx_h, idx_w = 0, 0
    H_canvas, W_canvas = 16, 16
    H_crop, W_crop = 2, 2

    img_CxHxW = torch.zeros((3, H_canvas, W_canvas))

    left, right, up, bottom = get_idxs_crop4(idx_h, idx_w, H_canvas, W_canvas, H_crop, W_crop)

    img_CxHxW[:, up:bottom, left:right] = torch.ones((3, H_crop, W_crop))

    img_CxHxW[:, idx_h, idx_w] = torch.tensor([1, 0, 1])

    plt_imgshow(img_CxHxW)
    plt_show()
