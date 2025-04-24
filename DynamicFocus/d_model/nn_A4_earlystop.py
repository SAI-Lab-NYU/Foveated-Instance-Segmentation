import numpy as np


class EarlyStopMin:
    def __init__(self):
        """
        初始化 EarlyStopping.
        :param patience: 当验证集性能不提升时，允许的最大连续 epoch 数.
        :param min_delta: 验证集性能提升的最小变化量.
        """
        self.loss_min = np.inf

    def check(self, val_loss):
        print(f"    \t{val_loss:.6} <? {self.loss_min:.6}")
        res = val_loss < self.loss_min
        self.loss_min = min(val_loss, self.loss_min)
        return res


class EarlyStopMax:
    def __init__(self):
        """
        初始化 EarlyStopping.
        :param patience: 当验证集性能不提升时，允许的最大连续 epoch 数.
        :param min_delta: 验证集性能提升的最小变化量.
        """
        self.loss_max = -np.inf

    def check(self, val_loss):
        print(f"    \t{val_loss:.6} >? {self.loss_max:.6}")
        res = val_loss > self.loss_max
        self.loss_max = max(val_loss, self.loss_max)
        return res
