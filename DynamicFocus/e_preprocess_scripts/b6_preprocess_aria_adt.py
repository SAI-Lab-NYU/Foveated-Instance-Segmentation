import os
import torch
from sympy.integrals.meijerint_doc import category
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm

import preset
from utility.fctn import save_json, save_jsonl, read_jsonl, load_image
from utility.plot_tools import plt_multi_imgshow, plt_show

fpath_adt_cache_infos_jsonl = os.path.join(preset.dpath_data_cache_AriaADT, 'smry.json')

K_fid = 'fid'  # frame index
K_ts = 'ts'  # timestamp
K_inm = 'inm'  # instance name
K_pnm = 'pnm'  # prototype name
K_cat = 'cat'  # category
K_iid = 'iid'  # instance id
K_cid = 'cid'  # category id
K_gz = 'gz'  # gaze
K_sp = 'sp'  # tensor shape
K_fpath = 'fpath' # png filepath


def fname2info(fname):
    body = fname.removesuffix('.pth.png')

    info = {}
    for sub in body.split('_'):
        bsidx = 1
        beidx = sub.find(']')
        key = sub[bsidx:beidx]
        value = sub[beidx + 1:]

        if key in ['fid', 'ts', 'iid', 'cid', 'gz', 'sp']:
            value = [int(v) for v in value.split('x')]

        info[key] = value
    return info


class DatasetADT(Dataset):
    """
    A template for creating custom datasets in PyTorch.
    """

    def __init__(self, refresh=True):
        self.HW_size = 1408
        if refresh:
            if os.path.exists(fpath_adt_cache_infos_jsonl):
                os.remove(fpath_adt_cache_infos_jsonl)

        if os.path.exists(fpath_adt_cache_infos_jsonl):
            self.infos = read_jsonl(fpath_adt_cache_infos_jsonl)

        else:
            self.infos = []
            for folder_seq in tqdm(os.listdir(preset.dpath_data_cache_AriaADT)):
                dpath_data = os.path.join(preset.dpath_data_cache_AriaADT, folder_seq, 'merged')
                for fname in os.listdir(dpath_data):
                    info = fname2info(fname)
                    info[K_fpath] = os.path.join(dpath_data, fname)
                    self.infos.append(info)

            save_jsonl(self.infos, fpath_adt_cache_infos_jsonl)

            cats = list({info[K_cat] for info in self.infos})
            prototypes = list({info[K_pnm] for info in self.infos})

            print(f"[prototypes] : {prototypes}")
            print(f"[prototypes] : {len(prototypes)} items")
            print(f"[cats] {cats}")
            print(f"[cats] {len(cats)} items")

            self.id2prototype = sorted(prototypes)
            self.prototype2id = {prototype: i for i, prototype in enumerate(self.id2prototype)}

            print(f"class_num K = {len(self.id2prototype)}")

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, idx):
        info = self.infos[idx]

        fpath = info[K_fpath]

        img_RGBS_4xHxW = load_image(fpath, mode='RGBA')

        img_RGB_3xHxW = img_RGBS_4xHxW[:3, :, :]
        seg_A_1xHxW = img_RGBS_4xHxW[3:, :, :]

        idx_H, idx_W = info[K_gz]

        flt_H = idx_H / (self.HW_size - 1)
        flt_W = idx_W / (self.HW_size - 1)

        F_HW_2 = torch.Tensor([flt_H, flt_W]).to(dtype=torch.float32)

        # float32 to int64
        prototype_idx = self.prototype2id[info[K_pnm]]

        Y_cls_1 = torch.Tensor([prototype_idx]).to(dtype=torch.int64)

        return img_RGB_3xHxW, F_HW_2, seg_A_1xHxW, Y_cls_1


if __name__ == '__main__':
    pass

    dataset = DatasetADT()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 遍历批量数据
    for batch in tqdm(dataloader):
        img_RGB_Bx3xHxW, F_HW_Bx2, seg_A_Bx1xHxW, Y_cls_Bx1 = batch
        print(f"img_RGB_3xHxW shape: {img_RGB_Bx3xHxW.shape} {img_RGB_Bx3xHxW.dtype}")
        print(f"F_HW_2 shape: {F_HW_Bx2.shape} {F_HW_Bx2.dtype}")
        print(f"seg_A_1xHxW shape: {seg_A_Bx1xHxW.shape} {seg_A_Bx1xHxW.dtype}")
        print(f"Y_cls_1 shape: {Y_cls_Bx1.shape} {Y_cls_Bx1.dtype}")
        break

    plt_multi_imgshow([img_RGB_Bx3xHxW[0], seg_A_Bx1xHxW[0]], row_col=(1, 2))
    plt_show()
