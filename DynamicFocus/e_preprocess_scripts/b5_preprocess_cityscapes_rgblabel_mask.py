import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from d_model.nn_A0_utils import try_gpu, try_cpu
from utility.plot_tools import *
from utility.watch import watch_time

from e_preprocess_scripts.a_preprocess_tools import CustomDataLoader, AbstractDataset
from utility.watch import Watch

import argparse
import traceback
from shapely import Polygon, Point

import numpy as np
import skimage
import torch
from tqdm import tqdm, trange

import preset
from utility.fctn import read_json, load_image, save_tensor, save_image, load_tensor, save_jsonl, read_jsonl, save_json
from utility.torch_tools import str_tensor_shape, add_alpha


def get_marker(N, marker_prefix):
    return f'{marker_prefix}{N}'


def wrap_name(name):
    return name.replace(' ', '-')


def clean_sample_folder(dataset_partition: str, marker: str):
    fdatasetname = 'cityscapes_rgblabel_mask'
    dpath_data_cook_lvis_part_marker = os.path.join(preset.dpath_data_cook, fdatasetname, dataset_partition, marker)

    dpath = dpath_data_cook_lvis_part_marker
    if os.path.exists(dpath):
        for fname in tqdm(list(os.listdir(dpath))):
            os.remove(os.path.join(dpath, fname))
        os.rmdir(dpath)
    print(f"CLEAN {dpath}")


labels_valid = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    ('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
    ('motorcycle', 1, 17, 'vehicle', 7, True, False, (0, 0, 230)),
    ('rider', 2, 12, 'human', 6, True, False, (255, 0, 0)),
    ('traffic light', 3, 6, 'object', 3, False, False, (250, 170, 30)),
    ('bus', 4, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    ('train', 5, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    ('truck', 6, 14, 'vehicle', 7, True, False, (0, 0, 70)),
    ('bicycle', 7, 18, 'vehicle', 7, True, False, (119, 11, 32)),
    ('traffic sign', 8, 7, 'object', 3, False, False, (220, 220, 0)),
    ('wall', 9, 3, 'construction', 2, False, False, (102, 102, 156)),
    ('fence', 10, 4, 'construction', 2, False, False, (190, 153, 153)),
    ('terrain', 11, 9, 'nature', 4, False, False, (152, 251, 152)),
    ('person', 12, 11, 'human', 6, True, False, (220, 20, 60)),
    ('pole', 13, 5, 'object', 3, False, False, (153, 153, 153)),
    ('sky', 14, 10, 'sky', 5, False, False, (70, 130, 180)),
    ('sidewalk', 15, 1, 'flat', 1, False, False, (244, 35, 232)),
    ('car', 16, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    ('vegetation', 17, 8, 'nature', 4, False, False, (107, 142, 35)),
    ('building', 18, 2, 'construction', 2, False, False, (70, 70, 70)),
    ('road', 19, 0, 'flat', 1, False, False, (128, 64, 128))
    # (  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    # (  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    # (  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    # (  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    # (  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    # (  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    # (  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    # (  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    # (  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    # (  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    # (  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]


class PreprocessCityscapesRGBLabelMask:

    @watch_time
    def __init__(self, dataset_partition='train'):

        self.dataset_partition = dataset_partition

        if dataset_partition == 'train':
            self.dpath_data_raw_cityscape_X = os.path.join(preset.dpath_data_raw_cityscape_X, 'train')
            self.dpath_data_raw_cityscape_Y = os.path.join(preset.dpath_data_raw_cityscape_Y, 'train')
        elif dataset_partition == 'valid':
            self.dpath_data_raw_cityscape_X = os.path.join(preset.dpath_data_raw_cityscape_X, 'val')
            self.dpath_data_raw_cityscape_Y = os.path.join(preset.dpath_data_raw_cityscape_Y, 'val')

        self.fdatasetname = 'cityscapes_rgblabel_mask'
        self.crop_H = 1024
        self.crop_W = 2048
        self.max_height = 1024
        self.max_width = 2048
        self.K = 20
        self.LABEL_unlabeled = 'unlabeled'

        self.path_data_cache_cityscapes = os.path.join(preset.dpath_data_cache, self.fdatasetname)
        self.path_data_cook_cityscapes = os.path.join(preset.dpath_data_cook, self.fdatasetname)
        self.path_data_cache_cityscapes_part = os.path.join(self.path_data_cache_cityscapes, self.dataset_partition)
        self.path_data_cook_cityscapes_part = os.path.join(self.path_data_cook_cityscapes, self.dataset_partition)

        os.makedirs(self.path_data_cache_cityscapes, exist_ok=True)
        os.makedirs(self.path_data_cook_cityscapes, exist_ok=True)
        os.makedirs(self.path_data_cache_cityscapes_part, exist_ok=True)
        os.makedirs(self.path_data_cook_cityscapes_part, exist_ok=True)

        self.idx2label, self.label2idx = self.get_labels_ordered()

        self.itemkey_fpathX_fpathY_s = self.cache_itemkey_fpathX_fpathY_s(use_cache=True)

    def get_labels_ordered(self):
        idx2label = [row[0] for row in labels_valid]
        idx2label = [wrap_name(label) for label in idx2label]
        label2idx = {label: idx for idx, label in enumerate(idx2label)}

        print(f"#idx2label : {len(idx2label)}")
        print(f"#label2idx : {len(label2idx)}")

        return idx2label, label2idx

    def cache_itemkey_fpathX_fpathY_s(self, use_cache=True):
        fname = 'itemkey_fpathX_fpathY_s'

        fpath = os.path.join(self.path_data_cache_cityscapes_part, f"{fname}.json")
        if use_cache and os.path.isfile(fpath):
            print(f"CACHE {fpath}")
            return read_json(fpath)
        else:

            key2fpath_X = {}
            for root, dirs, files in os.walk(self.dpath_data_raw_cityscape_X):
                for file in files:
                    fpath_png = os.path.join(root, file)
                    if fpath_png.endswith('.png'):
                        key = file.split('_leftImg8bit')[0]
                        key2fpath_X[key] = fpath_png
                        # print(key, fpath_png)

            key2fpath_Y = {}
            for root, dirs, files in os.walk(self.dpath_data_raw_cityscape_Y):
                for file in files:
                    fpath_png = os.path.join(root, file)
                    if fpath_png.endswith('_gtFine_color.png'):
                        key = file.split('_gtFine_color.png')[0]
                        key2fpath_Y[key] = fpath_png
                        # print(key, fpath_png)

            # Sanity Check
            keys = set(key2fpath_X.keys()) & set(key2fpath_Y.keys())

            itemkey_fpathX_fpathY_s = []
            count_missing = 0
            for key in keys:
                if key in key2fpath_X and key in key2fpath_Y:
                    bind = [key, key2fpath_X[key], key2fpath_Y[key]]
                    itemkey_fpathX_fpathY_s.append(bind)
                else:
                    print(f"{key} not in both feature or label")

            if count_missing == 0:
                print(f"Sanity Check Passed !")

            save_json(itemkey_fpathX_fpathY_s, fpath)
            print(f"SAVE {fpath}")

            return itemkey_fpathX_fpathY_s

    def get_objs_ordered(self, fpath_Y):
        info = read_json(fpath_Y)
        objs = info['objects']
        objs.sort(key=lambda obj: (
            self.label2idx[wrap_name(obj['label'])],
            len(obj['polygon'])
        )
                  )

        # important & small object at front
        return objs

    def make_a_sample(self, itemkey, fpath_X, fpath_Y, mark: str = 'default'):
        mapping = [[torch.tensor(row[-1]).to(dtype=torch.uint8, device='cuda'), torch.tensor(row[1]).to(dtype=torch.uint8, device='cuda')] for row in labels_valid]

        hsidx = self.max_height // 2 - self.crop_H // 2
        heidx = self.max_height // 2 + self.crop_H // 2
        wsidx = self.max_width // 2 - self.crop_W // 2
        weidx = self.max_width // 2 + self.crop_W // 2

        viewX_rgb_3xHSxWS_float32 = load_image(fpath_X)[:, hsidx:heidx, wsidx:weidx].to(device='cuda')
        viewX_rgb_3xHSxWS_uint8 = (viewX_rgb_3xHSxWS_float32 * 255).to(dtype=torch.uint8)
        viewY_rgb_3xHSxWS_float32 = (load_image(fpath_Y)[:, hsidx:heidx, wsidx:weidx]).to(device='cuda')
        viewY_rgb_3xHSxWS_uint8 = (viewY_rgb_3xHSxWS_float32 * 255.0).to(dtype=torch.uint8)

        Y_bind_1xHSxWS = torch.zeros((1, self.crop_H, self.crop_W), dtype=torch.uint8, device='cuda')

        for i in range(1, len(mapping)):
            color = mapping[i][0]  # 形状为 (3,)
            label = mapping[i][1]  # 标量

            # 扩展 color 的形状以进行广播比较
            # color[:, None, None]: 形状为 (3, 1, 1)
            # viewY_rgb_3xHSxWS: 形状为 (3, H, W)
            # 进行逐元素比较，得到形状为 (3, H, W) 的布尔张量
            mask = (viewY_rgb_3xHSxWS_uint8 == color[:, None, None])

            # 在通道维度 (dim=0) 上取逻辑与，得到形状为 (H, W) 的布尔张量
            mask_all = mask.all(dim=0)

            # 在 y_bind_1xHxW 中对应的位置填入标签值
            Y_bind_1xHSxWS[0, mask_all] = label

        sent_itemkey = f"{itemkey.replace('_', '-')}"
        sent_csize = f"{self.crop_H}x{self.crop_W}"

        fname_Y = f"{sent_itemkey}_1x{sent_csize}.uint8.Y.pt"
        fname_X = f"{sent_itemkey}_3x{sent_csize}.uint8.X.pt"

        fname_VX = f"{sent_itemkey}_3x{sent_csize}.float32.VX.png"
        fname_VY = f"{sent_itemkey}_3x{sent_csize}.float32.VY.png"

        save_tensor(Y_bind_1xHSxWS, os.path.join(self.path_data_cook_cityscapes_part, mark, fname_Y))
        save_tensor(viewX_rgb_3xHSxWS_uint8, os.path.join(self.path_data_cook_cityscapes_part, mark, fname_X))

        save_image(viewX_rgb_3xHSxWS_float32, os.path.join(self.path_data_cook_cityscapes_part, mark, fname_VX))
        save_image(viewY_rgb_3xHSxWS_float32, os.path.join(self.path_data_cook_cityscapes_part, mark, fname_VY))

    def make_N_samples(self, marker):

        dpath = os.path.join(self.path_data_cook_cityscapes_part, marker)
        os.makedirs(dpath, exist_ok=True)

        # 使用 ThreadPoolExecutor 实现多线程
        max_workers = max(1, os.cpu_count() - 1)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.make_a_sample, itemkey, fpathX, fpathY, marker)
                for itemkey, fpathX, fpathY in self.itemkey_fpathX_fpathY_s
            ]

            # 监控进度并捕获错误
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing samples"):
                try:
                    future.result()  # 如果任务出错，这里会抛出异常
                except Exception as e:
                    print(f"Error processing sample: {e}")

        print(f"make samples to {dpath}")


class DatasetCityScapesRGBLabelMask(AbstractDataset):

    def __init__(self, marker, dataset_partition='train'):
        super().__init__()

        self.K = len(labels_valid)
        self.cropH = 512
        self.cropW = 512
        self.K = 20

        self.marker = marker
        self.dataset_partition = dataset_partition

        if dataset_partition == 'train':
            self.dpath_data_raw_cityscape_X = os.path.join(preset.dpath_data_raw_cityscape_X, 'train')
            self.dpath_data_raw_cityscape_Y = os.path.join(preset.dpath_data_raw_cityscape_Y, 'train')
        elif dataset_partition == 'valid':
            self.dpath_data_raw_cityscape_X = os.path.join(preset.dpath_data_raw_cityscape_X, 'val')
            self.dpath_data_raw_cityscape_Y = os.path.join(preset.dpath_data_raw_cityscape_Y, 'val')

        self.fdatasetname = 'cityscapes_rgblabel_mask'

        self.path_data_cache_cityscapes = os.path.join(preset.dpath_data_cache, self.fdatasetname)
        self.path_data_cook_cityscapes = os.path.join(preset.dpath_data_cook, self.fdatasetname)
        self.path_data_cache_cityscapes_part = os.path.join(self.path_data_cache_cityscapes, self.dataset_partition)
        self.path_data_cook_cityscapes_part = os.path.join(self.path_data_cook_cityscapes, self.dataset_partition)

        os.makedirs(self.path_data_cache_cityscapes, exist_ok=True)
        os.makedirs(self.path_data_cook_cityscapes, exist_ok=True)
        os.makedirs(self.path_data_cache_cityscapes_part, exist_ok=True)
        os.makedirs(self.path_data_cook_cityscapes_part, exist_ok=True)

        self.path_data_cook_cityscapes_part_mark = os.path.join(self.path_data_cook_cityscapes_part, self.marker)

        self.fnames_Ypt = self.get_fnames_Ypt()
        self.idx2label, self.label2idx = self.get_labels_ordered()

    def get_fnames_Ypt(self):
        return [fname for fname in os.listdir(self.path_data_cook_cityscapes_part_mark) if fname.endswith('.Y.pt')]

    def get_namekeys(self):
        return self.fnames_Ypt

    def __len__(self) -> int:
        return len(self.fnames_Ypt)

    def __getitem__(self, index):
        fname_Y = self.fnames_Ypt[index]
        itemkey, IxHxW_Y = fname_Y.split('.')[0].split('_')

        IxHxW_X = '3x' + IxHxW_Y[2:]
        fname_X = f"{itemkey}_{IxHxW_X}.uint8.X.pt"

        fpath_Y = os.path.join(self.path_data_cook_cityscapes_part_mark, fname_Y)
        fpath_X = os.path.join(self.path_data_cook_cityscapes_part_mark, fname_X)

        Y_1xHxW = load_tensor(fpath_Y).to(dtype=torch.uint8)  # uint8 [0,19]
        X_3xHxW = load_tensor(fpath_X).to(dtype=torch.uint8)  # float32  [0.0,1.0]

        return X_3xHxW, Y_1xHxW

    def get_labels_ordered(self):
        idx2label = [row[0] for row in labels_valid]
        idx2label = [wrap_name(label) for label in idx2label]
        label2idx = {label: idx for idx, label in enumerate(idx2label)}

        print(f"#idx2label : {len(idx2label)}")
        print(f"#label2idx : {len(label2idx)}")

        return idx2label, label2idx


sp_default = 'sp0'


class DataLoaderCityScapesRGBLabelMask:

    def __init__(self, dataset: DatasetCityScapesRGBLabelMask):

        self.dataset = dataset

        self.max_height = 1024
        self.max_width = 2048
        self.cropH = self.dataset.cropH
        self.cropW = self.dataset.cropW
        self.K = self.dataset.K

        self.N = 15000 if self.dataset.dataset_partition == 'train' else 3000

    def get_iterator(self, batch_size, device: str = None, shuffle=True, xrange=range):

        N = self.N * 2
        B = len(self.dataset)
        base_array = np.arange(B)
        # 重复数组并截取到长度 N
        idxs = np.tile(base_array, (N // B) + 1)[:N]

        if shuffle:
            idxs = np.random.permutation(idxs)

        A = self.N if self.dataset.dataset_partition == 'train' else self.N
        for sidx in range(0, A, batch_size):
            items = []
            for offset_idx in range(batch_size):
                X_3xHxW, Y_1xHxW = self.dataset[idxs[sidx + offset_idx]]

                N_points = 0
                non_zero_coords = None
                X_3xHSxWS = None
                Y_1xHSxWS = None
                while N_points == 0 or non_zero_coords is None:
                    top = np.random.randint(0, self.max_height - self.cropH + 1)
                    left = np.random.randint(0, self.max_width - self.cropW + 1)

                    X_3xHSxWS = X_3xHxW[:, top:top + self.cropH, left:left + self.cropW]
                    Y_1xHSxWS = Y_1xHxW[:, top:top + self.cropH, left:left + self.cropW].clone()

                    non_zero_coords = torch.argwhere(Y_1xHSxWS[0] != 0)
                    N_points = len(non_zero_coords)

                selected_point = non_zero_coords[np.random.randint(0, N_points)]
                f_hidx, f_widx = selected_point[0], selected_point[1]

                target_cidx = Y_1xHSxWS[0, f_hidx, f_widx]
                Y_1xHSxWS = (Y_1xHSxWS == target_cidx).to(dtype=torch.uint8)

                F_Bx2 = torch.Tensor([f_hidx / self.cropH, f_widx / self.cropW]).to(dtype=torch.float32)
                Y_cls_Bx1 = torch.Tensor([target_cidx])

                bind = X_3xHSxWS, F_Bx2, Y_1xHSxWS, Y_cls_Bx1
                items.append(bind)

            data = [torch.stack(part, dim=0).to(device=device) for part in zip(*items)]
            X_Bx3xHSxWS, F_Bx2, Y_Bx1xHSxWS, Y_cls_Bx1 = data

            X_Bx3xHSxWS = X_Bx3xHSxWS.to(dtype=torch.float32) / 255.0
            F_Bx2 = F_Bx2.to(dtype=torch.float32)
            Y_Bx1xHSxWS = Y_Bx1xHSxWS.to(dtype=torch.float32)
            Y_cls_Bx1 = Y_cls_Bx1.to(dtype=torch.int64)

            # print(f"X_Bx3xHSxWS {X_Bx3xHSxWS.shape}")
            # print(f"F_Bx2 {F_Bx2.shape}")
            # print(f"Y_Bx1xHSxWS {Y_Bx1xHSxWS.shape}")
            # print(f"Y_cls_Bx1 {Y_cls_Bx1.shape}")
            # X_Bx3xHSxWS = F.interpolate(X_Bx3xHSxWS,(256,256),mode='nearest')
            # Y_Bx1xHSxWS = F.interpolate(Y_Bx1xHSxWS,(256,256),mode='nearest')

            yield X_Bx3xHSxWS, F_Bx2, Y_Bx1xHSxWS, Y_cls_Bx1


def get_Skwargs():
    parser = argparse.ArgumentParser(description="Script to process dataset with specified arguments.")

    # 添加参数
    parser.add_argument('--task', type=str, required=False, default='', help='Dataset partition to use (e.g., preprocess, speed_test )')

    # for preprocess
    parser.add_argument('--dataset_partition', nargs='+', type=str, required=False, help='Dataset partition to use (e.g., train, val, test)')
    parser.add_argument('--marker_prefix', type=str, required=False, default='sp', help='Marker to use for labeling or identification')

    # for speed_test
    parser.add_argument('--epoch', type=int, required=False, default=1, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, required=False, default=64, help='Size of each training batch')
    parser.add_argument('--target_device', type=str, required=False, default='gpu', help='Device to use for training (e.g., cpu, gpu)')
    parser.add_argument('--cache', action='store_true', help='Whether to cache the dataset in memory for faster access')
    parser.add_argument('--show', action='store_true', help='show sample number')

    parser.add_argument('--delete', action='store_true', help='delete cook data')
    parser.add_argument('--all_cidxs', action='store_true', help='delete cook data')

    # 解析参数
    return parser.parse_args()


def print_sample_count():
    dpath_train = os.path.join(preset.dpath_data_cook, 'cityscapes_rgblabel_mask', 'train')
    dpath_valid = os.path.join(preset.dpath_data_cook, 'cityscapes_rgblabel_mask', 'valid')

    for dpath in [dpath_train, dpath_valid]:
        print(dpath)
        for folder in os.listdir(dpath):
            dpath_sp = os.path.join(dpath, folder)
            print(folder, '\t', len(os.listdir(dpath_sp)))


if __name__ == '__main__':
    pass
    # ppcc_train = PreprocessCityscapes(dataset_partition='train')
    # ppcc_valid = PreprocessCityscapes(dataset_partition='valid')

    # ppcc_train.make_N_samples(100, marker='sp100')

    Skwargs = get_Skwargs()

    if Skwargs.task in ['preprocess', 'speed_test']:
        if Skwargs.dataset_partition is None:
            raise ValueError("Please specify dataset_partition.")

    if Skwargs.task == 'preprocess':

        w = Watch()
        ppcc_train = PreprocessCityscapesRGBLabelMask(dataset_partition='train')
        ppcc_valid = PreprocessCityscapesRGBLabelMask(dataset_partition='valid')

        clean_sample_folder(dataset_partition='train', marker=sp_default)
        clean_sample_folder(dataset_partition='valid', marker=sp_default)

        if 'train' in Skwargs.dataset_partition:
            ppcc_train.make_N_samples(marker=sp_default)
        if 'valid' in Skwargs.dataset_partition:
            ppcc_valid.make_N_samples(marker=sp_default)

        print(f"preprocess done! total cost {w.see_timedelta()}")

    elif Skwargs.task == 'speed_test':

        epoch = Skwargs.epoch
        batch_size = Skwargs.batch_size
        target_device = try_gpu() if Skwargs.target_device == 'gpu' else try_cpu()

        for dp in Skwargs.dataset_partition:
            w = Watch()

            datasetCityscapes = DatasetCityScapesRGBLabelMask(sp_default, dataset_partition=dp)

            dataloader = DataLoaderCityScapesRGBLabelMask(datasetCityscapes)
            print(f"CustomDataLoader Cache={Skwargs.cache} init {w.see_timedelta()}")

            for eidx in trange(epoch):
                for bidx, bparts in enumerate(dataloader.get_iterator(batch_size=batch_size, device=target_device, shuffle=False)):
                    if eidx == 0:
                        print()
                        for bpart in bparts:
                            print(bpart.device, bpart.dtype, str_tensor_shape(bpart))

                        X_Bx3xHSxWS, F_Bx2, Y_Bx1xHSxWS, Y_cls_Bx1 = bparts
                        print(Y_cls_Bx1.detach().cpu().numpy())
                        # plt_imgshow(X_Bx3xHSxWS[0])
                        # plt_imgshow(Y_Bx1xHSxWS[0])
                        #
                        # plt_show()

            print(f"CustomDataLoader Cache={Skwargs.cache} per epoch cost {w.see_timedelta() / epoch}")

            print(f"CustomDataLoader Cache={Skwargs.cache} total {w.total_timedelta()}")

    if Skwargs.show:
        print_sample_count()

    if Skwargs.delete:

        for dp in Skwargs.dataset_partition:
            for sp_train in Skwargs.sample_num:
                marker = get_marker(sp_train, Skwargs.marker_prefix)
                clean_sample_folder(dataset_partition=dp, marker=marker)
        print_sample_count()

"""

python e_preprocess_scripts/b5_preprocess_cityscapes_rgblabel_mask.py --task preprocess --dataset_partition train valid --sample_num 200

on local
python e_preprocess_scripts/b5_preprocess_cityscapes_rgblabel_mask.py --task preprocess --dataset_partition train valid



python e_preprocess_scripts/b5_preprocess_cityscapes_rgblabel_mask.py --task speed_test --dataset_partition train --epoch 5 --batch_size 25 --target_device gpu --cache
python e_preprocess_scripts/b5_preprocess_cityscapes_rgblabel_mask.py --task speed_test --dataset_partition train --epoch 5 --batch_size 25 --target_device gpu



# on server

python e_preprocess_scripts/b5_preprocess_cityscapes_rgblabel_mask.py --task preprocess --dataset_partition train valid --sample_num 3000
fg %1  # 或者 fg %2



ls -l | grep ^- | wc -l

cd  /home/hongyiz/DriverD/b_data_train/data_c_cook/cityscapes/train/


python e_preprocess_scripts/b5_preprocess_cityscapes_rgblabel_mask.py --show

python e_preprocess_scripts/b5_preprocess_cityscapes_rgblabel_mask.py --delete --dataset_partition train valid --sample_num 10 50 100 500 2500

"""
