import os
import random
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from d_model.nn_A0_utils import try_gpu, try_cpu
from utility.plot_tools import *
from utility.watch import watch_time
from multiprocessing import shared_memory
from e_preprocess_scripts.a_preprocess_tools_parallel import CustomDataLoader, AbstractDataset
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
torch.manual_seed(0)
np.random.seed(0)

def get_marker(N, marker_prefix):
    return f'{marker_prefix}{N}'


def get_rrcc_by_polygon(polygon, canvas_H, canvas_W):
    polygon_xy_Nx2 = np.array(polygon, dtype=np.int32)
    polygon_y_N = polygon_xy_Nx2[:, 0]
    polygon_x_N = polygon_xy_Nx2[:, 1]
    polygon_x_N[polygon_x_N < 0] = canvas_H + polygon_x_N[polygon_x_N < 0]
    polygon_y_N[polygon_y_N < 0] = canvas_W + polygon_y_N[polygon_y_N < 0]
    polygon_x_N = np.clip(polygon_x_N, 0, canvas_H - 1)
    polygon_y_N = np.clip(polygon_y_N, 0, canvas_W - 1)
    rr, cc = skimage.draw.polygon(polygon_x_N, polygon_y_N)
    return rr, cc


def wrap_name(name):
    return name.replace(' ', '-')


def find_polygon(objs, point, idx2label, label2idx):
    labelidx2idxs = [[] for _ in range(len(idx2label))]
    for idx, obj in enumerate(objs):
        label = wrap_name(obj['label'])
        polygon = obj['polygon']
        is_inside = Polygon(polygon).contains(Point(point))
        if is_inside:
            labelidx2idxs[label2idx[label]].append(idx)

    for idxs in labelidx2idxs:
        if len(idxs) > 0:
            tidx = idxs[0]
            obj = objs[tidx]
            label = wrap_name(obj['label'])
            polygon = obj['polygon']
            return label, polygon, tidx

    return "", [], -1


def clean_sample_folder(dataset_partition: str, marker: str):
    fdatasetname = 'cityscapes'
    dpath_data_cook_lvis_part_marker = os.path.join(preset.dpath_data_cook, fdatasetname, dataset_partition, marker)

    dpath = dpath_data_cook_lvis_part_marker
    if os.path.exists(dpath):
        for fname in tqdm(list(os.listdir(dpath))):
            os.remove(os.path.join(dpath, fname))
        os.rmdir(dpath)
    print(f"CLEAN {dpath}")


def cache_oidxHxW(objs, itemkey, path_data_cache_cityscapes_part, use_cache=True, max_height=1024, max_width=2048):
    bits2dtype = {8: torch.uint8, 16: torch.int16, 32: torch.int32, 64: torch.int64}

    try:
        dirname = 'oidx_HxW_s'
        os.makedirs(os.path.join(path_data_cache_cityscapes_part, dirname), exist_ok=True)

        fpath_oidx_HxW = os.path.join(path_data_cache_cityscapes_part, dirname, f"{itemkey}.HxW.oidx.pth")
        fpath_oidxs_visible = os.path.join(path_data_cache_cityscapes_part, dirname, f"{itemkey}.visible.oidxs.json")

        if use_cache and os.path.isfile(fpath_oidx_HxW):
            # print(f"CACHE {fpath}")
            oidx_HxW = load_tensor(fpath_oidx_HxW)
            oidxs_visible = read_json(fpath_oidxs_visible)
            return oidx_HxW, oidxs_visible
        else:
            N_objs = len(objs)
            base = 8 if N_objs < 255 else 16
            dtype_target = bits2dtype[base]
            oidx_default = 2 ** base - 1

            oidx_HxW = torch.ones((max_height, max_width), dtype=dtype_target) * oidx_default
            for oidx, obj in enumerate(objs[::-1]):
                # from least to most important objs, mark from back to front

                polygon = obj['polygon']
                rr, cc = get_rrcc_by_polygon(polygon, max_height, max_width)
                oidx_HxW[rr, cc] = N_objs - 1 - oidx
            save_tensor(oidx_HxW, fpath_oidx_HxW)

            oidxs_visible = torch.unique(oidx_HxW.flatten()).detach().cpu().numpy().tolist()

            save_json(oidxs_visible, fpath_oidxs_visible)

            # print(f"SAVE {fpath}")
            return oidx_HxW, oidxs_visible
    except Exception as e:
        traceback.print_exc()
        print(f"ERROR : {itemkey}")
        base = 8
        dtype_target = bits2dtype[base]
        oidx_default = 2 ** base - 1

        return torch.ones((max_height, max_width), dtype=dtype_target) * oidx_default, []

labels_valid = ["person", "persongroup",
                "rider", "ridergroup",

                "bicycle", "bicyclegroup",
                "motorcycle", "motorcyclegroup",
                "car", "cargroup",
                "truck", "truckgroup",
                "bus", "train",
                "caravan", "trailer",

                "license-plate",

                "traffic-light", "traffic-sign",
                "pole", "polegroup",

                "bridge", "fence", "guard-rail", "tunnel", "building", "wall",
                "rail-track", "sidewalk", "parking", "road",
                "vegetation", "terrain", "ground",
                "dynamic", "static", "ego-vehicle", "sky"]


class PreprocessCityscapes:

    @watch_time
    def __init__(self, dataset_partition='train'):

        self.dataset_partition = dataset_partition

        if dataset_partition == 'train':
            self.dpath_data_raw_cityscape_X = os.path.join(preset.dpath_data_raw_cityscape_X, 'train')
            self.dpath_data_raw_cityscape_Y = os.path.join(preset.dpath_data_raw_cityscape_Y, 'train')
        elif dataset_partition == 'valid':
            self.dpath_data_raw_cityscape_X = os.path.join(preset.dpath_data_raw_cityscape_X, 'val')
            self.dpath_data_raw_cityscape_Y = os.path.join(preset.dpath_data_raw_cityscape_Y, 'val')

        self.fdatasetname = 'cityscapes'
        self.crop_H = 512
        self.crop_W = 1024
        self.max_height = 1024
        self.max_width = 2048
        self.K = 41
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
        self.labelidx2fpathidxs = self.batch_cache_oidxHxW_labelidx2fpathidxs(use_cache=True)

        # self.info = self.get_info_lvis()
        # self.annotations = self.info['annotations']
        # 
        # # 1270141 train annotations
        # # 244707 valid annotations
        # self.id2imginfo = {img['id']: img for img in self.info['images']}
        # self.id2catyinfo = {caty['id']: caty for caty in self.info['categories']}  # 1203

        # save_json(self.id2catyinfo, os.path.join(self.path_data_cache_lvis_part, 'cid2info.json'))

    def get_labels_ordered(self):
        # idx2label = [
        #     "motorcycle", 
        #     "rider", 
        #     "traffic-light", 
        #     "bus", 
        #     "train", 
        #     "truck",
        #     "bicycle", 
        #     "traffic-sign",
        #     "wall",
        #     "fence", 
        #     "terrain",
        #     "person", 
        #     "pole", 
        #     "sky",
        #     "sidewalk",
        #     "car",
        #     "vegetation",
        #     "building",
        #     "persongroup",
        #     "rider", 
        #     "ridergroup",
        #     "bicyclegroup",
        #     "motorcyclegroup",
        #     "cargroup",
        #     "truckgroup",
        #     "caravan", 
        #     "trailer",
        #     "license-plate",
        #     "polegroup",
        #     "bridge","guard-rail", "tunnel", 
        #     "rail track", "parking", "road",
        #     "ground",
        #     "dynamic", "static", "ego-vehicle", 
        #     "out-of-roi", "rectification border",
        #     "unlabeled"
        # ]
        idx2label = [
            "person", "persongroup",
            "rider", "ridergroup",

            "bicycle", "bicyclegroup",
            "motorcycle", "motorcyclegroup",
            "car", "cargroup",
            "truck", "truckgroup",
            "bus", "train",
            "caravan", "trailer",

            "license plate",

            "traffic light", "traffic sign",
            "pole", "polegroup",

            "bridge", "fence", "guard rail", "tunnel", "building", "wall",
            "rail track", "sidewalk", "parking", "road",
            "vegetation", "terrain", "ground",
            "dynamic", "static", "ego vehicle", "sky",
            "out of roi", "rectification border",
            "unlabeled"
        ]

        idx2label = [wrap_name(label) for label in idx2label]
        label2idx = {label: idx for idx, label in enumerate(idx2label)}

        print(f"#idx2label : {len(idx2label)}")
        print(f"#label2idx : {len(label2idx)}")
        print(label2idx)
        print(idx2label)

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
                    if fpath_png.endswith('.json'):
                        key = file.split('_gtFine_polygons')[0]
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

    def batch_cache_oidxHxW_labelidx2fpathidxs(self, use_cache=False):
        fname = 'labelidx2fpathidxs'

        fpath = os.path.join(self.path_data_cache_cityscapes_part, f"{fname}.jsonl")
        if use_cache and os.path.isfile(fpath):
            print(f"CACHE {fpath}")
            return read_jsonl(fpath)
        else:
            labelidx2pathidxs = [[] for _ in range(self.K)]
            for pathidx, (itemkey, fpath_X, fpath_Y) in enumerate(tqdm(self.itemkey_fpathX_fpathY_s)):
                objs = self.get_objs_ordered(fpath_Y)
                oidx_HxW, oidxs_visible = cache_oidxHxW(objs, itemkey, self.path_data_cache_cityscapes_part, use_cache=True, max_height=self.max_height, max_width=self.max_width)

                for oidx in oidxs_visible:
                    if oidx != 255:
                        obj = objs[oidx]
                        label = wrap_name(obj['label'])
                        labelidx = self.label2idx[label]
                        labelidx2pathidxs[labelidx].append(pathidx)
            save_jsonl(labelidx2pathidxs, fpath)
            print(f"SAVE {fpath}")
            return labelidx2pathidxs

    def make_a_sample(self, itemkey, idx_H, idx_W, fpath_X, fpath_Y, oidx_target, mark: str = 'default'):
        objs_ordered = self.get_objs_ordered(fpath_Y)

        obj_target = objs_ordered[oidx_target]
        label_target = wrap_name(obj_target['label'])
        labelidx_target = self.label2idx[label_target]

        labelidx_tracker_HxW = torch.zeros((self.max_height, self.max_width), dtype=torch.uint8)

        polygon_target = obj_target['polygon']
        rr_t, cc_t = get_rrcc_by_polygon(polygon_target, self.max_height, self.max_width)
        labelidx_tracker_HxW[rr_t, cc_t] = 1

        # print(f"idxH,idxW = {idx_H}x{idx_W}")
        h_sidx = idx_H + random.randint(-(self.crop_H - 1), 0)
        h_eidx = h_sidx + self.crop_H
        w_sidx = idx_W + random.randint(-(self.crop_W - 1), 0)
        w_eidx = w_sidx + self.crop_W
        # print(f"before h_sidx,h_eidx = {h_sidx} ~ {h_eidx}")
        # print(f"before w_sidx,w_eidx = {w_sidx} ~ {w_eidx}")

        # 高度范围修正
        if h_sidx < 0:
            h_sidx, h_eidx = 0, self.crop_H
        if h_eidx > self.max_height:
            h_eidx, h_sidx = self.max_height, self.max_height - self.crop_H
        # print(f"after h_sidx, h_eidx = {h_sidx} ~ {h_eidx}")
        # print(f"after w_sidx, w_eidx = {w_sidx} ~ {w_eidx}")

        # 宽度范围修正
        if w_sidx < 0:
            w_sidx, w_eidx = 0, self.crop_W
        if w_eidx > self.max_width:
            w_eidx, w_sidx = self.max_width, self.max_width - self.crop_W

        idx_CH = idx_H - h_sidx
        idx_CW = idx_W - w_sidx

        view_rgb_3xHxW = load_image(fpath_X)

        labelidx_HSxWS = labelidx_tracker_HxW[h_sidx:h_eidx, w_sidx:w_eidx]
        view_rgb_3xHSxWS = view_rgb_3xHxW[:, h_sidx:h_eidx, w_sidx:w_eidx]

        X_bind_3xHxW = (view_rgb_3xHSxWS * 255).to(dtype=torch.uint8)
        Y_bind_1xHxW = labelidx_HSxWS.unsqueeze(0).to(dtype=torch.uint8)

        label_target = wrap_name(obj_target['label'])
        labelidx_target = self.label2idx[label_target]

        sent_catyname = label_target
        sent_cid = f"c{labelidx_target}"
        sent_kid = f"k{labelidx_target}"

        sent_itemkey = f"{itemkey.replace('_', '-')}"
        sent_fpos = f"{idx_CH}x{idx_CW}"
        sent_csize = f"{self.crop_H}x{self.crop_W}"

        fname_Y = f"{sent_catyname}_{sent_cid}_{sent_kid}_{sent_itemkey}_{sent_fpos}_1x{sent_csize}.uint8.Y.pt"
        fname_X = f"{sent_catyname}_{sent_cid}_{sent_kid}_{sent_itemkey}_{sent_fpos}_3x{sent_csize}.uint8.X.pt"
        fname_S = f"{sent_catyname}_{sent_cid}_{sent_kid}_{sent_itemkey}_{sent_fpos}_4x{sent_csize}.uint8.S.png"

        save_tensor(X_bind_3xHxW, os.path.join(self.path_data_cook_cityscapes_part, mark, fname_X))
        save_tensor(Y_bind_1xHxW, os.path.join(self.path_data_cook_cityscapes_part, mark, fname_Y))

        view_rgba_4xHSxWS = add_alpha(view_rgb_3xHSxWS)

        # canvas_HSxWS = 0 => alpha = alpha_mask
        # canvas_HSxWS = 1 => alpha = 1

        alpha_mask = 0.1
        view_rgba_4xHSxWS[-1, :, :] = (1 - alpha_mask) * (labelidx_HSxWS.to(dtype=torch.float32)) + alpha_mask

        color_focus = [1, 0, 1]
        view_rgba_4xHSxWS[:-1, idx_CH, idx_CW] = torch.tensor(color_focus, dtype=torch.float32)
        save_image(view_rgba_4xHSxWS, os.path.join(self.path_data_cook_cityscapes_part, mark, fname_S))

    def rank_itemkeys(self):
        itemkeys = []
        itemkey_2_Kobjs = {}
        itemkey_2_Nobjs = {}
        for itemkey, fpath_X, fpath_Y in tqdm(self.itemkey_fpathX_fpathY_s):
            objs = self.get_objs_ordered(fpath_Y)
            Nobjs = len(objs)
            Kobjs = len({obj["label"] for obj in objs})

            itemkeys.append(itemkey)
            itemkey_2_Kobjs[itemkey] = Kobjs
            itemkey_2_Nobjs[itemkey] = Nobjs

        itemkeys.sort(key=lambda itemkey: (itemkey_2_Kobjs[itemkey], itemkey_2_Nobjs[itemkey]), reverse=True)
        return itemkeys

    def make_a_sample_by_label(self, label_target='person', mark: str = 'default'):
        labelidx_target = self.label2idx[label_target]
        if labelidx_target >= self.K - 1:
            print(f"DO NOT LET label_target = 'unlabeled'")
            return
        print(self.labelidx2fpathidxs)
        pathidxs = self.labelidx2fpathidxs[labelidx_target]
        if len(pathidxs) == 0:
            print(f'NOT FOUND label_target = {label_target}')
            return

        pathidx = random.choice(pathidxs)
        itemkey, fpath_X, fpath_Y = self.itemkey_fpathX_fpathY_s[pathidx]

        objs_ordered = self.get_objs_ordered(fpath_Y)
        oidx_HxW, oidxs_visible = cache_oidxHxW(objs_ordered, itemkey, self.path_data_cache_cityscapes_part, use_cache=True, max_height=self.max_height, max_width=self.max_width)

        oidxs_valid = [oidx for oidx, obj in enumerate(objs_ordered) if wrap_name(obj['label']) == label_target]

        oidx_target = -1
        idx_H, idx_W = -1, -1
        random.shuffle(oidxs_valid)
        for oidx_candidate in oidxs_valid:
            focus_grid_Nx2 = torch.argwhere(oidx_HxW == oidx_candidate)

            N, _ = focus_grid_Nx2.shape
            if N > 0:
                idx_H, idx_W = focus_grid_Nx2[np.random.randint(N)].tolist()
                oidx_target = oidx_candidate
                break
        if oidx_target == -1:
            print(f"ERROR : label_target = {label_target}, itemkey = {itemkey}")
            return
        self.make_a_sample(itemkey, idx_H, idx_W, fpath_X, fpath_Y, oidx_target, mark=mark)

    def make_N_samples(self, N, marker):
        labels = [
            "motorcycle", 
            "rider", 
            "traffic-light", 
            "bus", 
            "train", 
            "truck",
            "bicycle", 
            "traffic-sign",
            "wall",
            "fence", 
            "terrain",
            "person", 
            "pole", 
            "sky",
            "sidewalk",
            "car",
            "vegetation",
            "building",
            'road']

        dpath = os.path.join(self.path_data_cook_cityscapes_part, marker)
        os.makedirs(dpath, exist_ok=True)

        for n in trange(N):
            target_label = labels[n % len(labels)]
            self.make_a_sample_by_label(target_label, mark=marker)
        print(f"make samples to {dpath}")

original_idx_to_new_idx = {
    6: 0,
    2: 1,
    17:2,
    12: 3,
    13: 4,
    10: 5,
    4: 6,
    18: 7,
    26:8,
    22: 9,
    32: 10,
    0: 11,
    19: 12,
    37: 13,
    28: 14,
    8: 15,
    31: 16,
    25: 17,
    30: 18
}

def convert_index(original_index):
    return original_idx_to_new_idx.get(original_index, 0)  # Returns -1 if index is not found


class DatasetCityScapes(AbstractDataset):

    def __init__(self, marker, dataset_partition='train'):
        super().__init__()
        self.HC = 512
        self.WC = 1024
        self.K = len(labels_valid)

        self.marker = marker
        self.dataset_partition = dataset_partition

        if dataset_partition == 'train':
            self.dpath_data_raw_cityscape_X = os.path.join(preset.dpath_data_raw_cityscape_X, 'train')
            self.dpath_data_raw_cityscape_Y = os.path.join(preset.dpath_data_raw_cityscape_Y, 'train')
        elif dataset_partition == 'valid':
            self.dpath_data_raw_cityscape_X = os.path.join(preset.dpath_data_raw_cityscape_X, 'val')
            self.dpath_data_raw_cityscape_Y = os.path.join(preset.dpath_data_raw_cityscape_Y, 'val')

        self.fdatasetname = 'cityscapes'

        self.path_data_cache_cityscapes = os.path.join(preset.dpath_data_cache, self.fdatasetname)
        self.path_data_cook_cityscapes = os.path.join(preset.dpath_data_cook, self.fdatasetname)
        self.path_data_cache_cityscapes_part = os.path.join(self.path_data_cache_cityscapes, self.dataset_partition)
        self.path_data_cook_cityscapes_part = os.path.join(self.path_data_cook_cityscapes, self.dataset_partition)

        os.makedirs(self.path_data_cache_cityscapes, exist_ok=True)
        os.makedirs(self.path_data_cook_cityscapes, exist_ok=True)
        os.makedirs(self.path_data_cache_cityscapes_part, exist_ok=True)
        os.makedirs(self.path_data_cook_cityscapes_part, exist_ok=True)

        self.path_data_cook_cityscapes_part_mark = os.path.join(self.path_data_cook_cityscapes_part, self.marker)

        # self.fnames_Ypt = self.get_fnames_Ypt()
        self.fnames_Ypt = list(self.get_fnames_Ypt())

    def get_fnames_Ypt(self):
        for entry in os.scandir(self.path_data_cook_cityscapes_part_mark):
            if entry.name.endswith('.Y.pt') and entry.is_file():
                yield entry.name
        # return [fname for fname in os.listdir(self.path_data_cook_cityscapes_part_mark) if fname.endswith('.Y.pt')]

    def get_namekeys(self):
        return self.fnames_Ypt

    def __len__(self) -> int:
        return len(self.fnames_Ypt)

    def __getitem__(self, index):
        fname_Y = self.fnames_Ypt[index]
        caty, cid, kid, itemkey, fpos, IxHxW_Y = fname_Y.split('.')[0].split('_')
        IxHxW_X = '3x' + IxHxW_Y[2:]
        fname_X = f"{caty}_{cid}_{kid}_{itemkey}_{fpos}_{IxHxW_X}.uint8.X.pt"

        Y_cls_s = convert_index(int(kid[1:]))

        fpath_Y = os.path.join(self.path_data_cook_cityscapes_part_mark, fname_Y)
        fpath_X = os.path.join(self.path_data_cook_cityscapes_part_mark, fname_X)

        Y_1xHxW = load_tensor(fpath_Y).to(dtype=torch.float32)
        X_3xHxW = load_tensor(fpath_X).to(dtype=torch.float32) / 255.0

        idx_H, idx_W = [int(num) for num in fpos.split('x')]
        F_2 = torch.Tensor([idx_H / self.HC, idx_W / self.WC]).to(dtype=torch.float32)

        Y_cls_1 = torch.Tensor([Y_cls_s]).to(dtype=torch.int64)

        return X_3xHxW, F_2, Y_1xHxW, Y_cls_1

class DatasetCityScapes_Parallel(AbstractDataset):
    def __init__(self, marker, dataset_partition='train', preload=False):
        super().__init__()
        self.HC = 512
        self.WC = 1024
        self.K = len(labels_valid)
        self.marker = marker
        self.dataset_partition = dataset_partition
        self.preload = preload

        self.setup_paths()
        self.fnames_Ypt = list(self.get_fnames_Ypt())
        
        if self.preload:
            self.data_cache = self._create_shared_memory_cache()
    
    
    def get_namekeys(self):
        return self.fnames_Ypt

    def __len__(self) -> int:
        return len(self.fnames_Ypt)

    def setup_paths(self):
        if self.dataset_partition == 'train':
            self.dpath_data_raw_cityscape_X = os.path.join(preset.dpath_data_raw_cityscape_X, 'train')
            self.dpath_data_raw_cityscape_Y = os.path.join(preset.dpath_data_raw_cityscape_Y, 'train')
        elif self.dataset_partition == 'valid':
            self.dpath_data_raw_cityscape_X = os.path.join(preset.dpath_data_raw_cityscape_X, 'val')
            self.dpath_data_raw_cityscape_Y = os.path.join(preset.dpath_data_raw_cityscape_Y, 'val')
        
        self.fdatasetname = 'cityscapes'
        self.path_data_cache_cityscapes = os.path.join(preset.dpath_data_cache, self.fdatasetname)
        self.path_data_cook_cityscapes = os.path.join(preset.dpath_data_cook, self.fdatasetname)
        self.path_data_cache_cityscapes_part = os.path.join(self.path_data_cache_cityscapes, self.dataset_partition)
        self.path_data_cook_cityscapes_part = os.path.join(self.path_data_cook_cityscapes, self.dataset_partition)
        os.makedirs(self.path_data_cache_cityscapes, exist_ok=True)
        os.makedirs(self.path_data_cook_cityscapes, exist_ok=True)
        os.makedirs(self.path_data_cache_cityscapes_part, exist_ok=True)
        os.makedirs(self.path_data_cook_cityscapes_part, exist_ok=True)
        self.path_data_cook_cityscapes_part_mark = os.path.join(self.path_data_cook_cityscapes_part, self.marker)

    def get_fnames_Ypt(self):
        for entry in os.scandir(self.path_data_cook_cityscapes_part_mark):
            if entry.name.endswith('.Y.pt') and entry.is_file():
                yield entry.name

    def _load_data(self, index):
        fname_Y = self.fnames_Ypt[index]
        caty, cid, kid, itemkey, fpos, IxHxW_Y = fname_Y.split('.')[0].split('_')
        IxHxW_X = '3x' + IxHxW_Y[2:]
        fname_X = f"{caty}_{cid}_{kid}_{itemkey}_{fpos}_{IxHxW_X}.uint8.X.pt"

        Y_cls_s = convert_index(int(kid[1:]))

        fpath_Y = os.path.join(self.path_data_cook_cityscapes_part_mark, fname_Y)
        fpath_X = os.path.join(self.path_data_cook_cityscapes_part_mark, fname_X)

        Y_1xHxW = load_tensor(fpath_Y).to(dtype=torch.float32)
        X_3xHxW = load_tensor(fpath_X).to(dtype=torch.float32) / 255.0

        idx_H, idx_W = [int(num) for num in fpos.split('x')]
        F_2 = torch.Tensor([idx_H / self.HC, idx_W / self.WC]).to(dtype=torch.float32)
        Y_cls_1 = torch.Tensor([Y_cls_s]).to(dtype=torch.int64)

        return X_3xHxW, F_2, Y_1xHxW, Y_cls_1

    def _create_shared_memory_cache(self):
        cache = {}
        for idx, fname_Y in enumerate(self.fnames_Ypt):
            data = self._load_data(idx)
            data_numpy = [d.numpy() for d in data]
            shm = [shared_memory.SharedMemory(create=True, size=d.nbytes) for d in data_numpy]
            for s, d in zip(shm, data_numpy):
                buffer = np.ndarray(d.shape, dtype=d.dtype, buffer=s.buf)
                buffer[:] = d[:]
            cache[idx] = shm
        return cache

    def __getitem__(self, index):
        if self.preload:
            shm = self.data_cache[index]
            data = [torch.from_numpy(np.ndarray((s.size,), dtype=np.float32, buffer=s.buf)).reshape(self.get_data_shape(i)) for i, s in enumerate(shm)]
            return data
        else:
            return self._load_data(index)

    def get_data_shape(self, i):
        if i == 0:
            return (3, self.HC, self.WC)  
        elif i == 1:
            return (2,)
        elif i == 2:
            return (1, self.HC, self.WC)
        elif i == 3:
            return (1,)
    
    def __del__(self):
        if self.preload:
            for shm_list in self.data_cache.values():
                for shm in shm_list:
                    shm.close()
                    shm.unlink()

def get_Skwargs():
    parser = argparse.ArgumentParser(description="Script to process dataset with specified arguments.")

    # 添加参数
    parser.add_argument('--task', type=str, required=False, default='', help='Dataset partition to use (e.g., preprocess, speed_test )')

    # for preprocess
    parser.add_argument('--dataset_partition', nargs='+', type=str, required=False, help='Dataset partition to use (e.g., train, val, test)')
    parser.add_argument('--sample_num', type=int, nargs='+', required=False, help='Number of samples to process')
    parser.add_argument('--marker_prefix', type=str, required=False, default='sp', help='Marker to use for labeling or identification')

    # for speed_test
    parser.add_argument('--epoch', type=int, required=False, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, required=False, default=64, help='Size of each training batch')
    parser.add_argument('--target_device', type=str, required=False, default='gpu', help='Device to use for training (e.g., cpu, gpu)')
    parser.add_argument('--cache', action='store_true', help='Whether to cache the dataset in memory for faster access')
    parser.add_argument('--show', action='store_true', help='show sample number')

    parser.add_argument('--delete', action='store_true', help='delete cook data')
    parser.add_argument('--all_cidxs', action='store_true', help='delete cook data')

    # 解析参数
    return parser.parse_args()


def print_sample_count():
    dpath_train = os.path.join(preset.dpath_data_cook, 'cityscapes', 'train')
    dpath_valid = os.path.join(preset.dpath_data_cook, 'cityscapes', 'valid')

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
        if Skwargs.dataset_partition is None or Skwargs.sample_num is None:
            raise ValueError("Please specify dataset_partition and sample_num.")

    if Skwargs.task == 'preprocess':

        w = Watch()
        pplv_train = PreprocessCityscapes(dataset_partition='train')
        pplv_valid = PreprocessCityscapes(dataset_partition='valid')
        for sp_train in Skwargs.sample_num:
            sp_valid = sp_train // 5
            marker_train = get_marker(sp_train, Skwargs.marker_prefix)
            marker_valid = get_marker(sp_valid, Skwargs.marker_prefix)
            print('--------------------')
            print(sp_train)
            clean_sample_folder(dataset_partition='train', marker=marker_train)
            clean_sample_folder(dataset_partition='valid', marker=marker_valid)

            if 'train' in Skwargs.dataset_partition:
                pplv_train.make_N_samples(sp_train, marker=marker_train)
            if 'valid' in Skwargs.dataset_partition:
                pplv_valid.make_N_samples(sp_valid, marker=marker_valid)

        print(f"preprocess done! total cost {w.see_timedelta()}")

    elif Skwargs.task == 'speed_test':

        epoch = Skwargs.epoch
        batch_size = Skwargs.batch_size
        target_device = try_gpu() if Skwargs.target_device == 'gpu' else try_cpu()

        for dp in Skwargs.dataset_partition:
            for sp_train in Skwargs.sample_num:
                w = Watch()

                marker = get_marker(sp_train, Skwargs.marker_prefix)
                datasetCityscapes = DatasetCityScapes(marker, dataset_partition=dp)

                print(f"CustomDataLoader Cache={Skwargs.cache} init {w.see_timedelta()}")

                dataloader = CustomDataLoader(datasetCityscapes, cache=Skwargs.cache)

                for eidx in trange(epoch):
                    for bidx, bparts in enumerate(dataloader.get_iterator(batch_size=batch_size, device=target_device, shuffle=True)):
                        if eidx == 0 and bidx == 0:
                            print()
                            for bpart in bparts:
                                print(bpart.device, bpart.dtype, str_tensor_shape(bpart))

                            X_3xHxW, F_2, Y_1xHxW, Y_cls_1 = bparts
                            print(F_2)
                            print(Y_cls_1)
                            plt_imgshow(X_3xHxW[0, :])
                            plt_imgshow(Y_1xHxW[0, :])
                            plt_show()

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

python e_preprocess_scripts/b3_preprocess_cityscapes.py --task preprocess --dataset_partition train --sample_num 2070

on local
python e_preprocess_scripts/b3_preprocess_cityscapes.py --task preprocess --dataset_partition train valid --sample_num 10 50 100 50000


python e_preprocess_scripts/b3_preprocess_cityscapes.py --task preprocess --dataset_partition train valid --sample_num 100 500 1000

python e_preprocess_scripts/b3_preprocess_cityscapes.py --task speed_test --dataset_partition train --sample_num 500 --epoch 5 --batch_size 64 --target_device gpu --cache
python e_preprocess_scripts/b3_preprocess_cityscapes.py --task speed_test --dataset_partition train --sample_num 500 --epoch 5 --batch_size 64 --target_device gpu


# on server

python e_preprocess_scripts/b3_preprocess_cityscapes.py --task preprocess --dataset_partition train valid --sample_num 10 50 100 500
fg %1  # 或者 fg %2



ls -l | grep ^- | wc -l

cd  /home/hongyiz/DriverD/b_data_train/data_c_cook/cityscapes/train/


python e_preprocess_scripts/b3_preprocess_cityscapes.py --show

python e_preprocess_scripts/b3_preprocess_cityscapes.py --delete --dataset_partition train valid --sample_num 10 50 100 500 2500

"""