import torch
import torch.nn as nn


def area_balanced_mse(input_BxHxW: torch.Tensor, target_BxHxW: torch.Tensor):
    """
    Computes an area-balanced Mean Squared Error (MSE) loss, where the balance weights
    are derived from the target tensor. Areas with target 0 or 1 are weighted inversely
    proportional to their frequency.

    Args:
    - input_BxHxW: torch.Tensor of shape (B, H, W), the predicted values.
    - target_BxHxW: torch.Tensor of shape (B, H, W), the ground truth values (0 or 1).

    Returns:
    - loss: torch.Tensor, the computed area-balanced MSE loss.
    """
    # Ensure that the input and target are the same size
    assert input_BxHxW.shape == target_BxHxW.shape, "Input and target must have the same shape (B, H, W)"

    ones_BxHxW = (target_BxHxW >= 0.5)
    zeros_BxHxW = ~ones_BxHxW

    mse_BxHxW = (input_BxHxW - target_BxHxW) ** 2

    eps = 1e-6  # Small value to avoid division by zero

    mse_one_BxHxW = (mse_BxHxW * ones_BxHxW).sum(dim=[-2, -1]) / (ones_BxHxW.sum(dim=(-2, -1)) + eps)
    mse_zero_BxHxW = (mse_BxHxW * zeros_BxHxW).sum(dim=[-2, -1]) / (zeros_BxHxW.sum(dim=[-2, -1]) + eps)

    loss = torch.mean(0.5 * mse_one_BxHxW + 0.5 * mse_zero_BxHxW)
    return loss


class BMSELoss(nn.modules.loss._Loss):
    def __init__(self, *args, **kwargs):
        super(BMSELoss, self).__init__()

        self.bmse_loss = area_balanced_mse

    def forward(self, y_pred_ds_Bx1xHSxWS, ys_real_ds_Bx1xHSxWS):
        return self.bmse_loss(y_pred_ds_Bx1xHSxWS, ys_real_ds_Bx1xHSxWS)


def area_balanced_cosim(input_BxK: torch.Tensor, target_Bx1: torch.Tensor, class_num: int):
    """
    Compute an area-balanced cosine similarity loss.

    Args:
        input_BxK (torch.Tensor): Input tensor of shape (B, K) where B is the batch size and K is the number of classes.
        target_Bx1 (torch.Tensor): Target tensor of shape (B, 1) containing the class indices.
        class_num (int): The number of different classes.

    Returns:
        torch.Tensor: The computed loss value.

    The function calculates the cosine similarity loss in an area-balanced manner by iterating through each class.
    """

    eps = 1e-6  # Small value to avoid division by zero
    output_B = input_BxK.gather(1, target_Bx1)
    target_B = target_Bx1.squeeze(1)
    loss_s = []
    for k in range(class_num):
        isK_B = target_B == k
        if any(isK_B):
            k_loss = torch.sum(output_B[isK_B]) / (torch.sum(isK_B) + eps)
            loss_s.append(k_loss)

    # print(loss_s)
    res_loss = sum(loss_s) / (len(loss_s))

    return 1 - res_loss


class BCOSIMLoss(nn.modules.loss._Loss):
    def __init__(self, class_num, *args, **kwargs):
        super(BCOSIMLoss, self).__init__()

        self.bcosim_loss = area_balanced_cosim
        self.class_num = class_num

    def forward(self, input_BxK: torch.Tensor, target_Bx1: torch.Tensor):
        return self.bcosim_loss(input_BxK, target_Bx1, self.class_num)


class WCELoss(nn.modules.loss._Loss):
    def __init__(self, class_num, *args, **kwargs):
        super(WCELoss, self).__init__()

        self.bcosim_loss = nn.CrossEntropyLoss()
        self.class_num = class_num

    def forward(self, input_BxK: torch.Tensor, target_Bx1: torch.Tensor):
        return torch.sigmoid(self.bcosim_loss(input_BxK, target_Bx1.squeeze(1)))


if __name__ == '__main__':
    pass
    """
    B = 5
    C = 3
    H = 256
    W = 512
    downsample_factor = 4
    HS = H // downsample_factor
    WS = W // downsample_factor
    K = 41

    A = 4
    y_pred_BxAxCxHSxWS = torch.randn(B, A, C, HS, WS)
    y_real_BxAxCxHSxWS = torch.randn(B, A, C, HS, WS)

    ls = area_balanced_mse(y_pred_BxAxCxHSxWS, y_real_BxAxCxHSxWS)
    """

    ######

    input_BxK = torch.Tensor([[0.1, 0.2, 0.3, 0.4],
                              [0.5, 0.6, 0.7, 0.8],
                              [0.9, 1.0, 1.1, 1.2],
                              [1.3, 1.4, 1.5, 1.6],
                              [1.7, 1.8, 1.9, 2.0]]).to(dtype=torch.float32)

    input_BxK = torch.Tensor([[0, 0, 1, 0],
                              [0, 1, 0, 0],
                              [0, 0, 0, 1],
                              [0, 0, 1, 0],
                              [0, 0, 1, 0]
                              ]).to(dtype=torch.float32)

    input_BxK = torch.Tensor([[0, 0, 0, 1],
                              [0, 0, 1, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1],
                              [0, 0, 0, 1]
                              ]).to(dtype=torch.float32)

    # 定义一个索引张量 Bx1，这里选每行的某个特定列
    target_Bx1 = torch.Tensor([2, 1, 3, 2, 2]).unsqueeze(1).to(dtype=torch.int64)
    class_num = 4

    loss_fctn: BCOSIMLoss = BCOSIMLoss(4)

    ls_cosim = area_balanced_cosim(input_BxK, target_Bx1, 4)

    print(ls_cosim)
