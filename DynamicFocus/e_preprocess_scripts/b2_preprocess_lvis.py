import os
import subprocess
import sys

from torch.distributed import monitored_barrier

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

import argparse
from abc import abstractmethod
from pprint import pprint

import numpy as np
import skimage
import torch
from tqdm import tqdm, trange

import preset
from d_model.nn_A0_utils import try_gpu, try_cpu
from e_preprocess_scripts.a_preprocess_tools import AbstractDataset, CustomDataLoader
from utility.fctn import read_json, save_pickle, read_pickle, load_image, save_tensor, save_image, load_tensor, save_json
from utility.plot_tools import *
from utility.torch_tools import get_padding, str_tensor_shape
from utility.watch import watch_time, Watch
import torch.nn.functional as F

from itertools import zip_longest


def merge_alternate(list1, list2):
    result = []
    # 使用 zip_longest 进行交错合并，并去除 None 值
    for a, b in zip_longest(list1, list2):
        if a is not None:
            result.append(a)
        if b is not None:
            result.append(b)
    return result


def help_reordering(A):
    # reordering original dataset iteratively, similar to shuffle?
    # len(A)>3:
    if len(A) == 0:
        return A
    if len(A) == 1:
        return A
    elif len(A) == 2:
        return A[::-1]
    elif len(A) == 3:
        return [A[1], A[2], A[0]]
    elif len(A) == 4:
        return [A[2], A[1], A[3], A[0]]
    else:
        mid = len(A) // 2
        back = help_reordering(A[mid + 1:])
        front = help_reordering(A[:mid])

        return [A[mid]] + merge_alternate(back, front)


def reordering(A):
    sv = A[0]
    ev = A[-1]
    Amiddle = help_reordering(A[1:-1])

    return [Amiddle[0]] + [ev, sv] + Amiddle[1:]


# A = list(range(100))
# Ar = reordering(A)
# print(Ar)
# vis = [[v, i] for i, v in enumerate(Ar)]
# plt.scatter(*zip(*vis))
# plt_show()


def get_marker(N, marker_prefix):
    return f'{marker_prefix}{N}'


def get_rrcc_by_polygon(polygon, canvas_H, canvas_W):
    polygon_xy_Nx2 = np.array(polygon, dtype=np.int32)
    polygon_y_N = polygon_xy_Nx2[:, 0] # coordinate x
    polygon_x_N = polygon_xy_Nx2[:, 1] # coordinate y
    # adjust negative coordinates with canvas
    polygon_x_N[polygon_x_N < 0] = canvas_H + polygon_x_N[polygon_x_N < 0]
    polygon_y_N[polygon_y_N < 0] = canvas_W + polygon_y_N[polygon_y_N < 0]
    polygon_x_N = np.clip(polygon_x_N, 0, canvas_H - 1)
    polygon_y_N = np.clip(polygon_y_N, 0, canvas_W - 1)
    # get mask in the polygon region
    rr, cc = skimage.draw.polygon(polygon_x_N, polygon_y_N)
    return rr, cc


def wrap_name(name):
    # translate each string with _ () ti --
    return '-'.join([part for part in name.translate(str.maketrans('_ ()', '----')).split('-') if part])


def clean_sample_folder(dataset_partition: str, marker: str):
    # clean original datafolder
    fdatasetname = 'lvis'
    dpath_data_cook_lvis_part_marker = os.path.join(preset.dpath_data_cook, fdatasetname, dataset_partition, marker)

    dpath = dpath_data_cook_lvis_part_marker
    if os.path.exists(dpath):
        for fname in tqdm(list(os.listdir(dpath))):
            os.remove(os.path.join(dpath, fname))
        os.rmdir(dpath)
    print(f"CLEAN {dpath}")


cids_valid = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 41, 43, 44, 45, 47, 48, 50, 51, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
              69, 70, 71, 72, 73, 74, 75, 76, 77, 79, 80, 81, 83, 84, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 106, 107, 108, 109, 110, 111, 112, 114, 115, 116, 117, 118, 120, 121, 122, 123, 124,
              125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 141, 142, 143, 145, 146, 147, 148, 149, 150, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 164, 165, 166, 167, 168, 169, 170, 171, 172,
              173, 174, 175, 176, 177, 178, 180, 181, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 203, 204, 205, 206, 207, 208, 209, 211, 212, 213, 214, 216, 217, 218, 219, 220, 221,
              222, 223, 224, 225, 226, 227, 228, 229, 230, 232, 235, 236, 238, 239, 240, 241, 242, 243, 246, 248, 249, 251, 252, 253, 254, 255, 256, 259, 260, 261, 263, 264, 266, 267, 268, 271, 272, 273, 274, 276, 277, 278, 279, 280,
              282, 283, 284, 285, 286, 288, 289, 290, 291, 293, 294, 295, 296, 297, 298, 299, 303, 305, 306, 307, 308, 309, 310, 311, 312, 314, 315, 316, 317, 318, 319, 320, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 334,
              335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 372, 373, 375, 377, 378, 379, 380, 381, 383, 384, 385,
              386, 387, 389, 390, 391, 392, 393, 394, 395, 396, 398, 399, 400, 401, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 415, 416, 417, 418, 419, 421, 422, 423, 424, 425, 427, 428, 429, 430, 432, 433, 434, 435, 436,
              437, 438, 440, 441, 442, 443, 444, 445, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 459, 460, 461, 462, 463, 464, 465, 466, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 479, 480, 482, 483, 484, 485, 486,
              487, 488, 489, 490, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 504, 505, 506, 507, 509, 510, 511, 512, 513, 514, 515, 516, 517, 519, 520, 521, 522, 523, 524, 525, 526, 528, 529, 530, 531, 532, 534, 535, 536, 537,
              538, 539, 540, 541, 543, 544, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 569, 570, 572, 573, 576, 578, 579, 580, 581, 584, 586, 587, 588, 589, 590, 591, 592,
              593, 595, 596, 598, 599, 600, 601, 602, 603, 604, 605, 608, 609, 610, 611, 612, 613, 614, 615, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 641, 642,
              643, 644, 645, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 663, 666, 667, 668, 669, 670, 671, 673, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 691, 692, 693,
              694, 695, 696, 697, 698, 699, 700, 701, 703, 704, 705, 706, 707, 708, 709, 711, 713, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 728, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 744, 745,
              746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 770, 771, 773, 774, 775, 776, 777, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 793, 794,
              795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 810, 811, 812, 813, 814, 816, 817, 818, 819, 820, 821, 822, 825, 826, 827, 828, 829, 830, 831, 832, 833, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844,
              845, 846, 847, 848, 852, 854, 857, 859, 860, 861, 863, 865, 866, 867, 868, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881, 882, 885, 886, 887, 888, 889, 891, 892, 893, 895, 896, 897, 898, 899, 900, 901, 903,
              904, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917, 919, 920, 921, 922, 923, 924, 926, 927, 928, 929, 930, 932, 933, 934, 935, 936, 937, 938, 939, 940, 941, 942, 943, 944, 946, 947, 948, 949, 950, 951, 952,
              953, 954, 955, 957, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 970, 971, 973, 974, 976, 977, 978, 979, 980, 981, 982, 983, 984, 985, 986, 988, 989, 991, 992, 993, 994, 995, 996, 997, 999, 1000, 1001, 1002, 1004,
              1006, 1007, 1008, 1009, 1011, 1012, 1013, 1014, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1030, 1031, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042, 1043, 1044, 1045, 1046,
              1047, 1048, 1049, 1050, 1051, 1052, 1053, 1055, 1056, 1057, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1070, 1071, 1072, 1073, 1074, 1076, 1077, 1078, 1079, 1081, 1082, 1083, 1085, 1086, 1087, 1088,
              1089, 1090, 1091, 1092, 1093, 1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1120, 1121, 1122, 1123, 1125, 1127, 1128,
              1129, 1130, 1131, 1132, 1133, 1134, 1136, 1137, 1138, 1139, 1140, 1141, 1142, 1143, 1144, 1145, 1147, 1148, 1149, 1150, 1151, 1152, 1153, 1154, 1155, 1156, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1168, 1169, 1170,
              1171, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 1198, 1200, 1201, 1202, 1203}

original_idx_to_new_idx = {77: 0, 421: 1, 816: 2, 1115: 3, 225: 4, 173: 5, 1052: 6, 982: 7, 1021: 8, 1050: 9, 76: 10, 110: 11, 361: 12, 1097: 13, 631: 14, 422: 15, 378: 16, 687: 17, 1077: 18, 912: 19, 183: 20, 445: 21, 1019: 22, 818: 23, 496: 24, 817: 25, 3: 26, 698: 27, 703: 28, 766: 29, 1123: 30, 19: 31, 1202: 32, 1071: 33, 1042: 34, 694: 35, 296: 36, 569: 37, 1117: 38, 1064: 39, 961: 40, 350: 41, 461: 42, 169: 43, 1043: 44, 392: 45, 36: 46, 90: 47, 181: 48, 139: 49, 150: 50, 1139: 51, 271: 52, 80: 53, 828: 54, 346: 55, 748: 56, 923: 57, 143: 58, 351: 59, 804: 60, 793: 61, 1142: 62, 230: 63, 898: 64, 94: 65, 719: 66, 1037: 67, 713: 68, 1110: 69, 836: 70, 708: 71, 1133: 72, 589: 73, 1008: 74, 692: 75, 118: 76, 96: 77, 1060: 78, 61: 79, 498: 80, 595: 81, 968: 82, 277: 83, 947: 84, 387: 85, 50: 86, 1026: 87, 835: 88, 66: 89, 207: 90, 592: 91, 881: 92, 976: 93, 716: 94, 347: 95, 1190: 96, 154: 97, 1079: 98, 1177: 99}
def convert_index(original_index):
    return original_idx_to_new_idx.get(original_index, 0)  # Returns -1 if index is not found

class PreprocessLVIS:

    @watch_time
    def __init__(self, dataset_partition='train'):

        self.dataset_partition = dataset_partition

        # get corresponding folder for train/val dataset
        if dataset_partition == 'train':
            self.fpath_lvis = preset.fpath_data_raw_lvis_train
            self.dpath_coco = preset.dpath_data_raw_coco_train
        elif dataset_partition == 'valid':
            self.fpath_lvis = preset.fpath_data_raw_lvis_valid
            self.dpath_coco = preset.dpath_data_raw_coco_valid

        self.fdatasetname = 'lvis'

        self.path_data_cache_lvis = os.path.join(preset.dpath_data_cache, self.fdatasetname)
        self.path_data_cook_lvis = os.path.join(preset.dpath_data_cook, self.fdatasetname)
        self.path_data_cache_lvis_part = os.path.join(self.path_data_cache_lvis, self.dataset_partition)
        self.path_data_cook_lvis_part = os.path.join(self.path_data_cook_lvis, self.dataset_partition)

        os.makedirs(self.path_data_cache_lvis, exist_ok=True)
        os.makedirs(self.path_data_cook_lvis, exist_ok=True)
        os.makedirs(self.path_data_cache_lvis_part, exist_ok=True)
        os.makedirs(self.path_data_cook_lvis_part, exist_ok=True)

        # get annotations from cached json file
        self.info = self.get_info_lvis()
        self.annotations = self.info['annotations']

        # 1270141 train annotations
        # 244707 valid annotations
        self.id2imginfo = {img['id']: img for img in self.info['images']}
        self.id2catyinfo = {caty['id']: caty for caty in self.info['categories']}  # 1203

        self.max_height = self.get_max_height()
        self.max_width = self.get_max_width()

        # save_json(self.id2catyinfo, os.path.join(self.path_data_cache_lvis_part, 'cid2info.json'))

    def get_cids_monitored(self, take_num_class=None):
        cid2num = {}
        cid2area_total = {}

        for anoinfo in tqdm(self.annotations):
            cid = anoinfo['category_id']
            if cid in cids_valid:
                area = anoinfo['area']
                if cid not in cid2num:
                    cid2num[cid] = 0
                    cid2area_total[cid] = 0.0
                cid2num[cid] += 1
                cid2area_total[cid] += area

        cids = list(cid2num.keys())
        cid_num_area_s = []
        threshold = 1000

        for cid in tqdm(cids):
            num = cid2num[cid]
            area_total = cid2area_total[cid]
            area = round(area_total / num)
            if num > threshold:
                cid_num_area_s.append((cid, num, area))
        N = len(cid_num_area_s)
        # print(f"# valid cid = {N}")

        areas = [area for cid, num, area in cid_num_area_s]

        cid_num_area_s.sort(key=lambda cid_num_area: cid_num_area[2], reverse=True)

        for cid, num, area in cid_num_area_s:
            print(f"[{cid}] [{num}] [{self.id2catyinfo[cid]['name']}] [{area}]")

        idxs_reordering = reordering(list(range(N)))

        cid_num_area_s_reordered = [cid_num_area_s[idx] for idx in idxs_reordering]

        cid_num_area_s_reordered_take = cid_num_area_s_reordered[:take_num_class]
        take_cids = [cid for cid, num, area in cid_num_area_s_reordered_take]
        take_areas = [area for cid, num, area in cid_num_area_s_reordered_take]
        # print(take_cids)

        # plt.hist(areas, density=True, bins=50, alpha=0.25, label='total')
        # plt.hist(take_areas, density=True, bins=50, alpha=0.25, label='take')
        # plt.legend()
        # plt_show()

        cids_monitored = take_cids
        print(f"# cids_monitored = {len(cids_monitored)}")
        for cid in cids_monitored:
            print(f"[{cid}] [{self.id2catyinfo[cid]['name']}] ")

        return cids_monitored

    def get_max_height(self):
        # 640
        return max(imginfo['height'] for imginfo in self.info['images'])

    def get_max_width(self):
        # 640
        return max(imginfo['width'] for imginfo in self.info['images'])

    def get_info_lvis(self):

        fpath_pkl = self.fpath_lvis + '.pkl'
        info = {}
        if os.path.exists(fpath_pkl):
            info = read_pickle(fpath_pkl)
        else:
            info = read_json(self.fpath_lvis)
            save_pickle(info, fpath_pkl)
        return info

    def make_a_sample(self, anoid, mark: str = 'default', cidx2kidx={}):
        target_device = try_cpu()

        annotation = self.annotations[anoid]
        catyid = annotation['category_id']
        imageid = annotation['image_id']
        imginfo = self.id2imginfo[imageid]
        catyinfo = self.id2catyinfo[catyid]

        fname_img = str(os.path.basename(imginfo['coco_url']))
        fpath_img = os.path.join(self.dpath_coco, fname_img)
        # print(fpath_img)
        if not os.path.exists(fpath_img):
            for dpath_coco in [preset.dpath_data_raw_coco_train, preset.dpath_data_raw_coco_valid, preset.dpath_data_raw_coco_test]:
                fpath_img = os.path.join(dpath_coco, fname_img)
                # print(fpath_img)
                if os.path.exists(fpath_img):
                    break

        H = imginfo['height']
        W = imginfo['width']

        y_1xHxW = torch.zeros(1, H, W).to(dtype=torch.uint8, device=target_device)

        polygons = annotation['segmentation']

        rr_s = []
        cc_s = []
        for polygon in polygons:
            polygon_2N = np.round(np.array(polygon)).astype(np.int32)

            polygon_w_N = np.clip(polygon_2N[0::2], 0, W - 1)
            polygon_h_N = np.clip(polygon_2N[1::2], 0, H - 1)
            rr, cc = skimage.draw.polygon(polygon_h_N, polygon_w_N)
            rr_s.append(rr)
            cc_s.append(cc)

        rrs = np.concatenate(rr_s)
        ccs = np.concatenate(cc_s)
        y_1xHxW[0, rrs, ccs] = 1

        ridx_rrcc = np.random.randint(len(rrs))

        ridx_H = rrs[ridx_rrcc]
        ridx_W = ccs[ridx_rrcc]

        x_rgba_4xHxW = load_image(fpath_img, mode='RGBA')

        alpha_mask = 0.25
        x_rgba_4xHxW[-1, :, :] = (1 - alpha_mask) * (y_1xHxW) + alpha_mask
        x_rgba_4xHxW[:-1, ridx_H, ridx_W] = torch.tensor([1, 0, 1], dtype=torch.float32, device=target_device)

        pad_left, pad_right, pad_top, pad_bottom = get_padding(H, W, self.max_height, self.max_width)

        x_rgba_4xHPxWP = F.pad(x_rgba_4xHxW, (pad_left, pad_right, pad_top, pad_bottom))

        ridx_H += pad_top
        ridx_W += pad_left

        sent_catyname = wrap_name(catyinfo['name'])
        sent_cid = f"c{catyid}"
        sent_kid = f"k{cidx2kidx[catyid]}" if cidx2kidx else f"k{catyid - 1}"
        sent_aid = f"a{anoid}"
        sent_imgname = f"{fname_img.strip('.jpg')}"
        sent_fpos = f"{ridx_H}x{ridx_W}"
        sent_pad = f"{pad_left}x{pad_right}x{pad_top}x{pad_bottom}"
        sent_csize = f"{H}x{W}"

        fname_Y = f"{sent_catyname}_{sent_cid}_{sent_kid}_{sent_aid}_{sent_imgname}_{sent_fpos}_{sent_pad}_1x{sent_csize}.uint8.Y.pt"
        fname_S = f"{sent_catyname}_{sent_cid}_{sent_kid}_{sent_aid}_{sent_imgname}_{sent_fpos}_{sent_pad}_4x{sent_csize}.uint8.S.png"

        fpath_Y = os.path.join(self.path_data_cook_lvis_part, mark, fname_Y)
        fpath_S = os.path.join(self.path_data_cook_lvis_part, mark, fname_S)

        save_tensor(y_1xHxW, fpath_Y)
        save_image(x_rgba_4xHPxWP, fpath_S)

    @watch_time
    def make_N_samples(self, N, marker, cids_monitored=None):

        dpath = os.path.join(self.path_data_cook_lvis_part, marker)
        os.makedirs(dpath, exist_ok=True)
        if cids_monitored is None:
            cids_monitored = sorted(list(self.id2catyinfo.keys()))

        cidx2kidx = {cidx: kidx for kidx, cidx in enumerate(cids_monitored)}

        for n in trange(N):
            target_cidx = cids_monitored[n % len(cids_monitored)]

            select_aidxs = [aid for aid, anoinfo in enumerate(self.annotations) if anoinfo['category_id'] == target_cidx]
            aidx = np.random.choice(select_aidxs)
            for i in range(2):
                #print(i)
                self.make_a_sample(aidx, mark=marker, cidx2kidx=cidx2kidx)
        print(f"make samples to {dpath}")


class DatasetLVIS(AbstractDataset):

    def __init__(self, marker, dataset_partition='train'):
        self.HC = 640
        self.WC = 640
        self.K = 1203

        self.marker = marker
        self.dataset_partition = dataset_partition

        if dataset_partition == 'train':
            self.fpath_lvis = preset.fpath_data_raw_lvis_train
            self.dpath_coco = preset.dpath_data_raw_coco_train
        elif dataset_partition == 'valid':
            self.fpath_lvis = preset.fpath_data_raw_lvis_valid
            self.dpath_coco = preset.dpath_data_raw_coco_valid

        self.fdatasetname = 'lvis'

        self.path_data_cache_lvis = os.path.join(preset.dpath_data_cache, self.fdatasetname)
        self.path_data_cook_lvis = os.path.join(preset.dpath_data_cook, self.fdatasetname)
        self.path_data_cache_lvis_part = os.path.join(self.path_data_cache_lvis, self.dataset_partition)
        self.path_data_cook_lvis_part = os.path.join(self.path_data_cook_lvis, self.dataset_partition)

        os.makedirs(self.path_data_cache_lvis, exist_ok=True)
        os.makedirs(self.path_data_cook_lvis, exist_ok=True)
        os.makedirs(self.path_data_cache_lvis_part, exist_ok=True)
        os.makedirs(self.path_data_cook_lvis_part, exist_ok=True)

        self.path_data_cook_lvis_part_mark = os.path.join(self.path_data_cook_lvis_part, self.marker)

        self.fnames_Ypt = self.get_fnames_Ypt()

    def get_fnames_Ypt(self):
        return [fname for fname in os.listdir(self.path_data_cook_lvis_part_mark) if fname.endswith('.Y.pt')]

    def get_namekeys(self):
        return self.fnames_Ypt

    def __len__(self) -> int:
        return len(self.fnames_Ypt)

    def __getitem__(self, index):
        fname = self.fnames_Ypt[index]
        # ski_c964_k3_a32848_000000050482_439x377_0x0x80x80_1x480x640.uint8.Y.pt
        # caty: category, cid: id of c, kid: id of index
        # aid: id of annotation, imgid: id of image
        # fpos: feature position, 439, 377
        # paddings: pad_leftxpad, pad_rightxpad, top_xpad, bottompad
        caty, cid, kid, aid, imgid, fpos, paddings, IxHxW = fname.split('.')[0].split('_')
        Y_cls_s = int(kid[1:])

        pad_left, pad_right, pad_top, pad_bottom = [int(num) for num in paddings.split('x')]
        idx_H, idx_W = [int(num) for num in fpos.split('x')]

        fpath_Y = os.path.join(self.path_data_cook_lvis_part_mark, fname)

        fname_img = f"{imgid}.jpg"
        fpath_img = os.path.join(self.dpath_coco, fname_img)
        # print(fpath_img)
        if not os.path.exists(fpath_img):
            for dpath_coco in [preset.dpath_data_raw_coco_train, preset.dpath_data_raw_coco_valid, preset.dpath_data_raw_coco_test]:
                fpath_img = os.path.join(dpath_coco, fname_img)
                # print(fpath_img)
                if os.path.exists(fpath_img):
                    break

        # Y_1xHxW segmentation mask
        # X_4xHxW original 4 channels image
        Y_1xHxW = load_tensor(fpath_Y)
        X_4xHxW = load_image(fpath_img, mode='RGBA')

        # pad from original size to 640x640, with value 0
        Y_seg_1xHPxWP = F.pad(Y_1xHxW, (pad_left, pad_right, pad_top, pad_bottom)).to(dtype=torch.float32)
        X_4xHPxWP = F.pad(X_4xHxW, (pad_left, pad_right, pad_top, pad_bottom))

        # normalize the feature postion to 0-1 range
        F_2 = torch.Tensor([idx_H / self.HC, idx_W / self.WC]).to(dtype=torch.float32)

        # float32 to int64
        Y_cls_1 = torch.Tensor([Y_cls_s]).to(dtype=torch.int64)

        return X_4xHPxWP, F_2, Y_seg_1xHPxWP, Y_cls_1


def get_Skwargs():
    # python e_preprocess_scripts/b2_preprocess_lvis.py 
    # --task preprocess 
    # --dataset_partition train valid 
    # --sample_num 10 50 100 500
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
    dpath_train = os.path.join(preset.dpath_data_cook, 'lvis', 'train')
    dpath_valid = os.path.join(preset.dpath_data_cook, 'lvis', 'valid')

    for dpath in [dpath_train, dpath_valid]:
        print(dpath)
        for folder in os.listdir(dpath):
            dpath_sp = os.path.join(dpath, folder)
            print(folder, '\t', len(os.listdir(dpath_sp)))


if __name__ == '__main__':
    pass
    # pplv_train = PreprocessLVIS(dataset_partition='train')
    # pplv_train.get_cids_monitored(take_num_class=50)
    # pplv_valid = PreprocessLVIS(dataset_partition='valid')

    # cids = pplv_train.rank_cls_by_anum(take_num_class=25)

    #
    # shared_cidxs = set(pplv_train.id2catyinfo.keys()) & set(pplv_valid.id2catyinfo.keys())
    #
    # cidx2num_train = {cidx: 0 for cidx in tqdm(shared_cidxs)}
    # cidx2num_valid = {cidx: 0 for cidx in tqdm(shared_cidxs)}
    #
    # for aid, anoinfo in enumerate(tqdm(pplv_train.annotations)):
    #     cidx = anoinfo['category_id']
    #     cidx2num_train[cidx] += 1
    #
    # for aid, anoinfo in enumerate(tqdm(pplv_valid.annotations)):
    #     cidx = anoinfo['category_id']
    #     cidx2num_valid[cidx] += 1

    Skwargs = get_Skwargs()

    if Skwargs.task in ['preprocess', 'speed_test']:
        if Skwargs.dataset_partition is None or Skwargs.sample_num is None:
            raise ValueError("Please specify dataset_partition and sample_num.")

    if Skwargs.task == 'preprocess':

        w = Watch()
        pplv_train = PreprocessLVIS(dataset_partition='train')
        pplv_valid = PreprocessLVIS(dataset_partition='valid')

        for sp_train in Skwargs.sample_num:
            sp_valid = sp_train // 5
            marker_train = get_marker(sp_train, Skwargs.marker_prefix)
            marker_valid = get_marker(sp_valid, Skwargs.marker_prefix)
            # clean_sample_folder(dataset_partition='train', marker=marker_train)
            clean_sample_folder(dataset_partition='valid', marker=marker_valid)

            cids_monitored = None
            # if not Skwargs.all_cidxs:
            #     # makesure each class has enough samples
            #     class_num = int(round(max(5.0, sp_train * 1203 / 1270141)))

            #     cids_monitored = pplv_train.get_cids_monitored(take_num_class=class_num)
            
            cids_monitored = [77,
                                421,
                                816,
                                1115,
                                225,
                                173,
                                1052,
                                982,
                                1021,
                                1050,
                                76,
                                110,
                                361,
                                1097,
                                631,
                                422,
                                378,
                                687,
                                1077,
                                912,
                                183,
                                445,
                                1019,
                                818,
                                496,
                                817,
                                3,
                                698,
                                703,
                                766,
                                1123,
                                19,
                                1202,
                                1071,
                                1042,
                                694,
                                296,
                                569,
                                1117,
                                1064,
                                961,
                                350,
                                461,
                                169,
                                1043,
                                392,
                                36,
                                90,
                                181,
                                # 139,
                                # 150,
                                # 1139,
                                # 271,
                                # 80,
                                # 828,
                                # 346,
                                # 748,
                                # 923,
                                # 143,
                                # 351,
                                # 804,
                                # 793,
                                # 1142,
                                # 230,
                                # 898,
                                # 94,
                                # 719,
                                # 1037,
                                # 713,
                                # 1110,
                                # 836,
                                # 708,
                                # 1133,
                                # 589,
                                # 1008,
                                # 692,
                                # 118,
                                # 96,
                                # 1060,
                                # 61,
                                # 498,
                                # 595,
                                # 968,
                                # 277,
                                # 947,
                                # 387,
                                # 50,
                                # 1026,
                                # 835,
                                # 66,
                                # 207,
                                # 592,
                                # 881,
                                # 976,
                                # 716,
                                # 347,
                                # 1190,
                                # 154,
                                # 1079,
                                # 1177
                                ]

            '''
            run this command: python e_preprocess_scripts/b2_preprocess_lvis.py --task preprocess --dataset_partition train valid --sample_num 100000
            '''
            print(len(cids_monitored))


            if 'train' in Skwargs.dataset_partition:
                pplv_train.make_N_samples(sp_train, marker=marker_train, cids_monitored=cids_monitored)
            if 'valid' in Skwargs.dataset_partition:
                pplv_valid.make_N_samples(sp_valid, marker=marker_valid, cids_monitored=cids_monitored)

        print(f"preprocess done! total cost {w.see_timedelta()}")

    elif Skwargs.task == 'speed_test':

        epoch = Skwargs.epoch
        batch_size = Skwargs.batch_size
        target_device = try_gpu() if Skwargs.target_device == 'gpu' else try_cpu()

        for dp in Skwargs.dataset_partition:
            for sp_train in Skwargs.sample_num:
                w = Watch()

                marker = get_marker(sp_train, Skwargs.marker_prefix)
                datasetLVIS = DatasetLVIS(marker, dataset_partition=dp)

                print(f"CustomDataLoader Cache={Skwargs.cache} init {w.see_timedelta()}")

                dataloader = CustomDataLoader(datasetLVIS, cache=Skwargs.cache)

                for eidx in trange(epoch):
                    for bidx, bparts in enumerate(dataloader.get_iterator(batch_size=batch_size, device=target_device, shuffle=True)):
                        if eidx == 0 and bidx == 0:
                            print()
                            for bpart in bparts:
                                print(bpart.device, bpart.dtype, str_tensor_shape(bpart))

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

python e_preprocess_scripts/b2_preprocess_lvis.py --task preprocess --dataset_partition train --sample_num 2070

on local
python e_preprocess_scripts/b2_preprocess_lvis.py --task preprocess --dataset_partition train valid --sample_num 10 50 100 50000


python e_preprocess_scripts/b2_preprocess_lvis.py --task preprocess --dataset_partition train valid --sample_num 100 500 1000

python e_preprocess_scripts/b2_preprocess_lvis.py --task speed_test --dataset_partition train --sample_num 500 --epoch 5 --batch_size 64 --target_device gpu --cache
python e_preprocess_scripts/b2_preprocess_lvis.py --task speed_test --dataset_partition train --sample_num 500 --epoch 5 --batch_size 64 --target_device gpu


# on server

python e_preprocess_scripts/b2_preprocess_lvis.py --task preprocess --dataset_partition train valid --sample_num 10 1000
fg %1  # 或者 fg %2



ls -l | grep ^- | wc -l

cd  /home/hongyiz/DriverD/b_data_train/data_c_cook/lvis/train/


python e_preprocess_scripts/b2_preprocess_lvis.py --show

python e_preprocess_scripts/b2_preprocess_lvis.py --delete --dataset_partition train valid --sample_num 10 50 100 500 2500

"""
