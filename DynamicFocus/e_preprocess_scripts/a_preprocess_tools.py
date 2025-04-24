import os
import sys
from abc import ABC, abstractmethod

from d_model.nn_A0_utils import calc_tensor_memsize
from utility.watch import Watch

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from torch.utils.data import Dataset, DataLoader

import numpy as np
import torch


def get_random_point(H, W):
    # get rate in height and width dimension and get corresponding index
    rate_h = np.random.rand()
    rate_w = np.random.rand()
    idx_H = int(np.round(rate_h * (H - 1)))
    idx_W = int(np.round(rate_w * (W - 1)))
    return rate_h, rate_w, idx_H, idx_W


class AbstractDataset(Dataset, ABC):

    def __init__(self):
        super().__init__()
        self.dataset_partition = ''

    @abstractmethod
    def get_namekeys(self):
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass


class CustomDataLoader():

    def __init__(self, dataset: AbstractDataset, xrange=range, cache=False):
        # cache is used to prevent visiting the disk repeatly
        self.dataset = dataset
        self.cache = cache

        self.data = None
        if cache:
            w = Watch()
            items = []

            for idx in xrange(len(self.dataset)):
                items.append(self.dataset[idx])
            # concat data in a tensor list
            self.data = [torch.stack(part, dim=0) for part in zip(*items)]
            total_size = sum(calc_tensor_memsize(d) for d in self.data)
            print(f"cache cost {w.see_timedelta()} cache size = {total_size} MB")

    def get_iterator(self, batch_size, device: str = None, shuffle=True, xrange=range):
        idxs = np.arange(len(self.dataset))
        if shuffle:
            idxs = np.random.permutation(idxs)

        # simple to device and yield and reduce memory use
        if self.cache and (self.data is not None):
            for b in xrange(0, len(idxs), batch_size):
                batch_idxs = idxs[b:b + batch_size]

                parts = [part[batch_idxs].to(device=device) for part in self.data]
                yield parts

        else:
            for b in xrange(0, len(idxs), batch_size):

                batch_idxs = idxs[b:b + batch_size]

                items = []
                for idx in batch_idxs:
                    items.append(self.dataset[idx])

                parts = [torch.stack(part, dim=0).to(device=device) for part in zip(*items)]

                yield parts

    def delete(self):
        del self.data
        self.data = None


if __name__ == '__main__':
    pass
