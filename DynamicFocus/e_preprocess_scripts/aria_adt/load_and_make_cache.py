import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

from concurrent.futures.process import ProcessPoolExecutor

import shutil
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pprint import pprint

import cv2
import matplotlib.pyplot as plt
import pandas as pd

import preset
from utility.plot_tools import plt_imgshow, plt_show, plt_multi_imgshow
from utility.torch_tools import str_tensor_shape
from utility.xprint import pbox

from torch.nn import functional as F
import zipfile

import requests
from tqdm import tqdm
import torch

from e_preprocess_scripts.aria_adt.aria_const import fpath_index_json, dpath_cache_aria_adt
from utility.fctn import read_json, save_tensor, save_image, save_jsonl, read_jsonl, load_tensor, save_pickle, read_pickle
import numpy as np
import os
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.projects.adt import (
    AriaDigitalTwinDataProvider,
    AriaDigitalTwinSkeletonProvider,
    AriaDigitalTwinDataPathsProvider,
    bbox3d_to_line_coordinates,
    bbox2d_to_image_coordinates,
)


def download_file(url, save_path):
    try:
        # 发送 HTTP GET 请求
        response = requests.get(url, stream=True)
        response.raise_for_status()  # 检查请求是否成功

        # 将文件写入本地路径
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        print(f"文件已下载到: {save_path}")

    except requests.exceptions.RequestException as e:
        print(f"下载失败: {e}")


def unzip_file(zip_path, extract_folder):
    try:
        # 确保目标文件夹存在
        os.makedirs(extract_folder, exist_ok=True)

        # 解压 ZIP 文件
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)
        print(f"解压完成: {extract_folder}")


    except zipfile.BadZipFile:
        print(f"错误: 该文件不是有效的 ZIP 文件: {zip_path}")

    except FileNotFoundError:
        print(f"错误: 压缩包不存在: {zip_path}")

    except Exception as e:
        print(f"解压失败: {e}")


# def print_pth_files_info(folder_path):
#     """
#     遍历文件夹，读取所有 .pth 文件，并打印文件名、形状和数据类型。
#
#     参数:
#         folder_path (str): 文件夹路径。
#     """
#     # 遍历文件夹中的所有文件
#     for filename in os.listdir(folder_path):
#         # 检查文件是否为 .pth 文件
#         if filename.endswith(".pth"):
#             # 构建文件的完整路径
#             file_path = os.path.join(folder_path, filename)
#
#             # 加载 .pth 文件
#             data = torch.load(file_path)
#
#             # 打印文件名、形状和数据类型
#             if isinstance(data, torch.Tensor):  # 如果是张量
#                 print(f"文件名: {filename}, 形状: {data.shape}, 数据类型: {data.dtype}")
#             elif isinstance(data, dict):  # 如果是字典
#                 print(f"文件名: {filename}")
#                 for key, value in data.items():
#                     if isinstance(value, torch.Tensor):
#                         print(f"  Key: {key}, 形状: {value.shape}, 数据类型: {value.dtype}")
#                     else:
#                         print(f"  Key: {key}, 类型: {type(value)}")
#             else:  # 其他类型
#                 print(f"文件名: {filename}, 类型: {type(data)}")


def rgb_to_int(rgb_tensor):
    """
    将 (3, H, W) 的 RGB 张量转换为 (1, H, W) 的整型张量
    :param rgb_tensor: 输入形状为 (3, H, W) 的 RGB 张量
    :return: 输出形状为 (1, H, W) 的整型张量
    """
    # 拆分通道
    r, g, b = rgb_tensor[0], rgb_tensor[1], rgb_tensor[2]

    # 编码为整数
    int_tensor = (r.to(torch.int32) << 16) | (g.to(torch.int32) << 8) | b.to(torch.int32)

    # 添加新的维度，变成 (1, H, W)
    return int_tensor.unsqueeze(0)


ALPHA_fg_255 = 255
ALPHA_bg_128 = 128
ALPHA_focus_0 = 0


def process_box(input_tensor, gaze_x, gaze_y):
    """
    在指定的矩形框中找到数量最多的值，将该值设置为1，其余值设置为0
    :param input_tensor: 输入张量，形状为 (H, W)
    :param hmin: 矩形框的最小高度索引
    :param hmax: 矩形框的最大高度索引
    :param wmin: 矩形框的最小宽度索引
    :param wmax: 矩形框的最大宽度索引
    :return: 处理后的张量
    """
    # 截取矩形框区域
    """
    box_region = input_tensor[ymin:ymax, xmin:xmax]
    
    box_region_flattened = box_region.flatten(start_dim=0)
    # 找到矩形框区域中的唯一值和它们的频率
    unique_vals, counts = torch.unique(box_region_flattened, dim=0, return_counts=True)

    # print(sum(counts))
    # print(box_region_flattened.shape)
    # 找到数量最多的值（众数）
    mode_val = unique_vals[torch.argmax(counts)]

    # print(unique_vals, counts, mode_val)

    # 创建掩码，众数位置为1，其余为0
    mask = (box_region == mode_val).to(dtype=torch.uint8)

    # 创建一个与原张量相同的空张量并设置为0
    output_tensor = torch.zeros_like(input_tensor).to(dtype=torch.uint8)

    # 将掩码写回到原张量的指定位置
    output_tensor[ymin:ymax, xmin:xmax] = mask
    """

    mode_val = input_tensor[gaze_y, gaze_x]
    output_tensor = (input_tensor == mode_val).to(dtype=torch.uint8)

    # output_tensor = (ALPHA_fg_255 - ALPHA_bg_128) * output_tensor + ALPHA_bg_128

    return output_tensor


def find_min_max_coordinates_torch(mask):
    """
    Finds the minimum and maximum H (height) and W (width) indices in a 0-1 mask tensor where the value is 1.

    Parameters:
        mask (torch.Tensor): A 2D binary tensor (HxW) with values 0 and 1.

    Returns:
        tuple: (min_H, max_H, min_W, max_W) representing the bounds of the region where the mask is 1.
    """
    if not isinstance(mask, torch.Tensor):
        raise ValueError("Input must be a PyTorch tensor.")
    if mask.ndim != 2:
        raise ValueError("Input mask must be a 2D tensor.")

    # Find the indices where the mask is 1
    indices = torch.nonzero(mask == 1, as_tuple=False)

    if indices.size(0) == 0:
        # No 1s in the mask
        return None  # or (None, None, None, None)

    # Extract minimum and maximum H and W
    min_H, min_W = indices.min(dim=0).values
    max_H, max_W = indices.max(dim=0).values

    return int(min_H), int(max_H), int(min_W), int(max_W)


def sanitize_filename_simple(filename, replacement='-'):
    """
    Replaces illegal characters in a file name with a specified replacement.

    Parameters:
        filename (str): The input file name to sanitize.
        replacement (str): The character to replace illegal characters with (default is '-').

    Returns:
        str: The sanitized file name.
    """
    # List of illegal characters for file names
    illegal_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']

    # Replace each illegal character with the replacement
    for char in illegal_chars:
        filename = filename.replace(char, replacement)

    # Remove leading and trailing spaces
    filename = filename.strip()

    return filename


indexinfo = read_json(fpath_index_json)

sequences = indexinfo['sequences']
stream_id_str = "214-1"
stream_id = StreamId(stream_id_str)


def download_a_file(url, filename, dpath_seq):
    """
    Download and handle a single file, including optional unzip and cleanup.

    Parameters:
        url (str): The URL to download from.
        filename (str): The target file name.
        dpath_seq (str): The destination directory for the downloaded file.
    """
    fpath_filename = os.path.join(dpath_seq, filename)
    download_file(url, fpath_filename)
    if filename.endswith('.zip'):
        unzip_file(fpath_filename, extract_folder=dpath_seq)
        os.remove(fpath_filename)


def download_a_sequence(skey, worker=8):
    dpath_seq = os.path.join(dpath_cache_aria_adt, skey)
    os.makedirs(dpath_seq, exist_ok=True)

    filename_s = []
    url_s = []
    for subkey, info in sequences[skey].items():
        filename = info['filename']
        url = info['download_url']
        filename_s.append(filename)
        url_s.append(url)

    executor = ThreadPoolExecutor(max_workers=worker)

    futures = [
        executor.submit(download_a_file, url, filename, dpath_seq)
        for url, filename in zip(url_s, filename_s)
    ]

    # 显示进度条
    for future in tqdm(as_completed(futures), total=len(filename_s)):
        try:
            future.result()  # 等待任务完成
        except Exception as e:
            print(f"Error downloading sequence: {e}")
    executor.shutdown(wait=True)

    src_file = os.path.join(dpath_seq, f'ADT_{skey}_main_recording.vrs')
    copy_destination = os.path.join(dpath_seq, f'video.vrs')
    if os.path.exists(copy_destination):
        os.remove(copy_destination)
    os.rename(src_file, copy_destination)


def get_timestamp_ns(skey, filename='2d_bounding_box.csv', colkey='timestamp[ns]'):
    dpath_seq = os.path.join(dpath_cache_aria_adt, skey)
    os.makedirs(dpath_seq, exist_ok=True)
    fpath_csv = os.path.join(dpath_seq, filename)

    df = pd.read_csv(fpath_csv)

    df = df[df['stream_id'] == stream_id_str]

    ts_ns = df[colkey]

    res = list(sorted(set(ts_ns.tolist())))
    return res


def get_gt_provider(skey):
    dpath_seq = os.path.join(dpath_cache_aria_adt, skey)
    os.makedirs(dpath_seq, exist_ok=True)
    paths_provider = AriaDigitalTwinDataPathsProvider(dpath_seq)
    data_paths = paths_provider.get_datapaths()
    print("loading ground truth data...")
    gt_provider = AriaDigitalTwinDataProvider(data_paths)
    return gt_provider


def gen_seg_pth(skey, gt_provider, timestamp_ns_s):
    dpath_seg = os.path.join(dpath_cache_aria_adt, skey, 'seg_pth')
    os.makedirs(dpath_seg, exist_ok=True)

    for timestamp_ns in tqdm(timestamp_ns_s):
        seg_with_dt = gt_provider.get_segmentation_image_by_timestamp_ns(timestamp_ns, stream_id)
        seg_for_viz_raw_HxWxC = seg_with_dt.data().get_visualizable().to_numpy_array()
        tensor_seg_raw_HxWxC = torch.from_numpy(seg_for_viz_raw_HxWxC)
        # tensor_seg_rotated = tensor_seg_raw.permute(1, 0, 2).flip(1)  # rotate 90 degree
        #
        # tensor_seg_rotated_CxHxW = tensor_seg_rotated.permute(2, 0, 1)
        tensor_seg_rotated_CxHxW = tensor_seg_raw_HxWxC.permute(2, 0, 1)

        save_pickle(tensor_seg_rotated_CxHxW, os.path.join(dpath_seg, f"{timestamp_ns}.{str_tensor_shape(tensor_seg_rotated_CxHxW)}.pth.pkl"))
        # save_image(tensor_seg_rotated_CxHxW, os.path.join(dpath_seg, f"{timestamp_ns}.{str_tensor_shape(tensor_seg_rotated_CxHxW)}.png"))


def gen_img_pth(skey, gt_provider, timestamp_ns_s):
    dpath_img = os.path.join(dpath_cache_aria_adt, skey, 'img_pth')
    os.makedirs(dpath_img, exist_ok=True)

    for timestamp_ns in tqdm(timestamp_ns_s):

        synthetic_with_dt = gt_provider.get_synthetic_image_by_timestamp_ns(timestamp_ns, stream_id)

        if synthetic_with_dt.is_valid():
            synthetic_image = synthetic_with_dt.data().to_numpy_array()
            tensor_syn_raw_HxWxC = torch.from_numpy(synthetic_image)
            # tensor_syn_rotated = tensor_syn_raw.permute(1, 0, 2).flip(1)  # rotate 90 degree
            #
            # tensor_syn_rotated_CxHxW = tensor_syn_rotated.permute(2, 0, 1)
            tensor_syn_rotated_CxHxW = tensor_syn_raw_HxWxC.permute(2, 0, 1)

            save_pickle(tensor_syn_rotated_CxHxW, os.path.join(dpath_img, f"{timestamp_ns}.{str_tensor_shape(tensor_syn_rotated_CxHxW)}.pth.pkl"))
            # save_image(tensor_syn_rotated_CxHxW, os.path.join(dpath_img, f"{timestamp_ns}.{str_tensor_shape(tensor_syn_rotated_CxHxW)}.png"))
        else:
            print("synthetic image not valid for input timestamp!")


def gen_gaze_obj_json(skey, gt_provider, timestamp_ns_s):
    fpath_gaze_obj = os.path.join(dpath_cache_aria_adt, skey, 'gaze_obj.jsonl')
    gazes = []
    for i, timestamp_ns in tqdm(enumerate(timestamp_ns_s)):
        try:
            eye_gaze_with_dt = gt_provider.get_eyegaze_by_timestamp_ns(timestamp_ns)
            assert eye_gaze_with_dt.is_valid(), "Eye gaze not available"

            cam_calibration = gt_provider.get_aria_camera_calibration(stream_id)

            # Project the gaze center in CPF frame into camera sensor plane, with multiplication performed in homogenous coordinates
            eye_gaze = eye_gaze_with_dt.data()
            gaze_center_in_cpf = np.array([np.tan(eye_gaze.yaw), np.tan(eye_gaze.pitch), 1.0], dtype=np.float64) * eye_gaze.depth
            transform_cpf_sensor = gt_provider.raw_data_provider_ptr().get_device_calibration().get_transform_cpf_sensor(cam_calibration.get_label())
            gaze_center_in_camera = transform_cpf_sensor.inverse().to_matrix() @ np.hstack((gaze_center_in_cpf, 1)).T
            gaze_center_in_camera = gaze_center_in_camera[:3] / gaze_center_in_camera[3:]
            gaze_center_in_pixels = cam_calibration.project(gaze_center_in_camera)

            # print(gaze_center_in_pixels)

            bbox2d_with_dt = gt_provider.get_object_2d_boundingboxes_by_timestamp_ns(timestamp_ns, stream_id)

            # check if the result is valid
            if not bbox2d_with_dt.is_valid():
                print("2D bounding box is not available")
            # print("groundtruth_time - query_time = ", bbox2d_with_dt.dt_ns(), "ns")
            bbox2d_all_objects = bbox2d_with_dt.data()

            gaze_wh = [float(v) for v in gaze_center_in_pixels] if not gaze_center_in_pixels is None else None
            monitered_keys = ['instance id', 'instance name', 'prototype name', 'category', 'category uid']
            invalid_prototype_name_s = []
            xmin, xmax, ymin, ymax = 0, 0, 0, 0

            k2v_s = []

            if gaze_wh is not None:
                gaze_x, gaze_y = gaze_wh

                for obj_id, obj in bbox2d_all_objects.items():
                    xmin, xmax, ymin, ymax = obj.box_range

                    # if (xmin < gaze_x < xmax) and (ymin < gaze_y < ymax):
                    if gt_provider.has_instance_id(obj_id):
                        target_obj_info = gt_provider.get_instance_info_by_id(obj_id)
                        sents = [sent for sent in str(target_obj_info).split('\n')]
                        k2v = {}
                        for sent in sents:
                            coma_idx = sent.find(':')
                            k = sent[:coma_idx]
                            v = sent[coma_idx + 2:]
                            if k in monitered_keys:
                                k2v[k] = int(v) if k in ['instance id', 'category uid'] else v

                        k2v['bbox2d'] = [float(xmin), float(xmax), float(ymin), float(ymax)]
                        k2v['visibility_ratio'] = obj.visibility_ratio
                        if not (k2v['prototype name'] in invalid_prototype_name_s):
                            k2v_s.append(k2v)

            gaze = {'fidx': i,
                    'gaze': gaze_wh,
                    'obj_num': len(k2v_s),
                    'obj_infos': k2v_s,
                    'ts_ns': int(timestamp_ns)
                    }

            # print(gaze)
            gazes.append(gaze)

        except Exception as e:
            print(traceback.format_exc())
    save_jsonl(gazes, fpath_gaze_obj)


def merge_img_seg_gaze_obj_unit(skey, gaze_obj):
    dpath_seq_merged = os.path.join(dpath_cache_aria_adt, skey, 'merged')
    os.makedirs(dpath_seq_merged, exist_ok=True)

    dpath_seq_debug = os.path.join(dpath_cache_aria_adt, skey, 'debug')
    os.makedirs(dpath_seq_debug, exist_ok=True)

    dpath_img = os.path.join(dpath_cache_aria_adt, skey, 'img_pth')
    os.makedirs(dpath_img, exist_ok=True)

    dpath_seg = os.path.join(dpath_cache_aria_adt, skey, 'seg_pth')
    os.makedirs(dpath_seg, exist_ok=True)

    fidx = gaze_obj['fidx']
    gaze = gaze_obj['gaze']
    obj_infos = gaze_obj['obj_infos']
    obj_num = gaze_obj['obj_num']
    ts_ns = gaze_obj['ts_ns']
    if gaze:

        gaze_x, gaze_y = gaze
        gaze_x, gaze_y = int(round(gaze_x)), int(round(gaze_y))

        seg_rgb_tensor_3xHxW = read_pickle(os.path.join(dpath_seg, f"{int(ts_ns)}.3x1408x1408.pth.pkl"))
        img_rgb_tensor_3xHxW = read_pickle(os.path.join(dpath_img, f"{int(ts_ns)}.3x1408x1408.pth.pkl"))

        seg_int_tensor_1xHxW = rgb_to_int(seg_rgb_tensor_3xHxW)
        mask_int_tensor_HxW = process_box(seg_int_tensor_1xHxW[0], gaze_x, gaze_y)

        if obj_num:

            for obj_info in obj_infos:
                bbox2d = obj_info['bbox2d']
                xmin, xmax, ymin, ymax = bbox2d

                xmin, xmax, ymin, ymax = int(round(xmin)), min(int(round(xmax)), 1407), int(round(ymin)), min(int(round(ymax)), 1407)
                final_mask_int_tensor_HxW = torch.zeros_like(mask_int_tensor_HxW).to(dtype=torch.uint8)
                final_mask_int_tensor_HxW[ymin:ymax, xmin:xmax] = mask_int_tensor_HxW[ymin:ymax, xmin:xmax]

                obj_info['fg_count'] = 0
                obj_info['box2d_area'] = -0
                if final_mask_int_tensor_HxW[gaze_y, gaze_x]:
                    box_region = final_mask_int_tensor_HxW[ymin:ymax, xmin:xmax].to(dtype=torch.int64).flatten(start_dim=0)
                    fg_count = box_region.sum().item()
                    total_count = len(box_region)

                    if xmin <= gaze_x <= xmax and ymin <= gaze_y <= ymax:

                        if total_count != 0:
                            cut_ymin, cut_ymax, cut_xmin, cut_xmax = find_min_max_coordinates_torch(final_mask_int_tensor_HxW)

                            # print(xmin, xmax, ymin, ymax)
                            # print(cut_ymin, cut_ymax, cut_xmin, cut_xmax)
                            obj_info['bbox2d_cut'] = [cut_xmin, cut_xmax, cut_ymin, cut_ymax]
                            obj_info['bbox2d_diff'] = sum(np.abs(np.array(obj_info['bbox2d']) - np.array(obj_info['bbox2d_cut'])))
                            obj_info['fg_count'] = fg_count
                            obj_info['box2d_area'] = total_count

            obj_infos = [obj_info for obj_info in obj_infos if (obj_info['fg_count'] > 0) and ('bbox2d_diff' in obj_info)]

            min_bbox2d_diff = min([obj_info['bbox2d_diff'] for obj_info in obj_infos])
            obj_infos = [obj_info for obj_info in obj_infos if obj_info['bbox2d_diff'] == min_bbox2d_diff]

            if len(obj_infos) > 0:
                obj_infos.sort(key=lambda x: x['fg_count'], reverse=True)
                # print()
                # print(f"fidx{fidx}")
                # pbox(obj_infos)

                cur_obj_info = obj_infos[0]

                bbox2d = cur_obj_info['bbox2d']
                xmin, xmax, ymin, ymax = bbox2d

                xmin, xmax, ymin, ymax = int(round(xmin)), min(int(round(xmax)), 1407), int(round(ymin)), min(int(round(ymax)), 1407)

                cur_final_mask_int_tensor_HxW = torch.zeros_like(mask_int_tensor_HxW).to(dtype=torch.uint8)
                cur_final_mask_int_tensor_HxW[ymin:ymax, xmin:xmax] = mask_int_tensor_HxW[ymin:ymax, xmin:xmax]

                debug_cur_final_mask_int_tensor_HxW = (ALPHA_fg_255 - ALPHA_bg_128) * cur_final_mask_int_tensor_HxW + ALPHA_bg_128

                debug_int_tensor_4xHxW = torch.cat([img_rgb_tensor_3xHxW, debug_cur_final_mask_int_tensor_HxW.unsqueeze(0)], dim=0)

                debug_int_tensor_4xHxW = debug_int_tensor_4xHxW.permute(0, 2, 1)
                gaze_h, gaze_w = gaze_x, gaze_y
                hmin, hmax, wmin, wmax = xmin, xmax, ymin, ymax

                debug_int_tensor_4xHxW[:, :, gaze_w] = torch.tensor([255, 0, 255, 255])[:, None]
                debug_int_tensor_4xHxW[:, gaze_h, :] = torch.tensor([255, 0, 255, 255])[:, None]

                debug_int_tensor_4xHxW[:, hmin, wmin:wmax] = torch.tensor([255, 0, 0, 255], dtype=torch.uint8)[:, None]
                debug_int_tensor_4xHxW[:, hmax, wmin:wmax] = torch.tensor([255, 0, 0, 255], dtype=torch.uint8)[:, None]
                debug_int_tensor_4xHxW[:, hmin:hmax, wmin] = torch.tensor([255, 0, 0, 255], dtype=torch.uint8)[:, None]
                debug_int_tensor_4xHxW[:, hmin:hmax:, wmax] = torch.tensor([255, 0, 0, 255], dtype=torch.uint8)[:, None]

                inst_id = cur_obj_info['instance id']
                inst_name = cur_obj_info['instance name']
                proto_name = cur_obj_info['prototype name']
                category = sanitize_filename_simple(cur_obj_info['category'])
                category_uid = cur_obj_info['category uid']

                # fidx, ts_ns,
                fname_body = f"""[fid]{str(fidx).zfill(4)}_[ts]{ts_ns}_[inm]{inst_name}_[pnm]{proto_name}_[cat]{category}_[iid]{inst_id}_[cid]{category_uid}_[gz]{gaze_h}x{gaze_w}_[sp]{str_tensor_shape(debug_int_tensor_4xHxW)}"""
                fname_png = f"""{fname_body}.png"""
                fname_pth_png = f"""{fname_body}.pth.png"""

                plt_multi_imgshow([debug_int_tensor_4xHxW, seg_rgb_tensor_3xHxW.permute(0, 2, 1), cur_final_mask_int_tensor_HxW.permute(1, 0)], row_col=[2, 2])
                plt.savefig(os.path.join(dpath_seq_debug, fname_png))
                plt.close('all')

                debug_int_tensor_4xHxW = torch.cat([img_rgb_tensor_3xHxW.permute(0, 2, 1), cur_final_mask_int_tensor_HxW.permute(1, 0).unsqueeze(0) * 255], dim=0)

                save_image(debug_int_tensor_4xHxW, os.path.join(dpath_seq_merged, fname_pth_png))


def merge_img_seg_gaze_obj_batch(skey, worker=8):
    dpath_seq_merged = os.path.join(dpath_cache_aria_adt, skey, 'merged')
    os.makedirs(dpath_seq_merged, exist_ok=True)

    dpath_seq_debug = os.path.join(dpath_cache_aria_adt, skey, 'debug')
    os.makedirs(dpath_seq_debug, exist_ok=True)

    dpath_img = os.path.join(dpath_cache_aria_adt, skey, 'img_pth')
    os.makedirs(dpath_img, exist_ok=True)

    dpath_seg = os.path.join(dpath_cache_aria_adt, skey, 'seg_pth')
    os.makedirs(dpath_seg, exist_ok=True)

    fpath_gaze_obj = os.path.join(dpath_cache_aria_adt, skey, 'gaze_obj.jsonl')

    gaze_obj_infos = read_jsonl(fpath_gaze_obj)
    executor = ProcessPoolExecutor(max_workers=8)
    # 提交任务
    futures = [
        executor.submit(merge_img_seg_gaze_obj_unit, skey, gaze_obj)
        for gaze_obj in gaze_obj_infos
    ]

    # 收集结果，并在 tqdm 中显示进度
    for future in tqdm(as_completed(futures), total=len(gaze_obj_infos)):
        try:
            # 如果需要返回值，可以通过 `future.result()` 获取
            result = future.result()
        except Exception as e:
            print(f"Error processing gaze_obj: {e}")
    executor.shutdown(wait=True)


def clean(skey):
    dpath_seq_merged = os.path.join(dpath_cache_aria_adt, skey, 'merged')
    os.makedirs(dpath_seq_merged, exist_ok=True)

    dpath_seq_debug = os.path.join(dpath_cache_aria_adt, skey, 'debug')
    os.makedirs(dpath_seq_debug, exist_ok=True)

    dpath_img = os.path.join(dpath_cache_aria_adt, skey, 'img_pth')
    os.makedirs(dpath_img, exist_ok=True)

    dpath_seg = os.path.join(dpath_cache_aria_adt, skey, 'seg_pth')
    os.makedirs(dpath_seg, exist_ok=True)

    fpath_gaze_obj = os.path.join(dpath_cache_aria_adt, skey, 'gaze_obj.jsonl')

    keep_paths = [dpath_seq_merged, dpath_seq_debug, fpath_gaze_obj]

    dpath_seq = os.path.join(dpath_cache_aria_adt, skey)
    for fname in tqdm(list(os.listdir(dpath_seq))):
        fpath = os.path.join(dpath_seq, fname)
        if not fpath in keep_paths:
            try:
                if os.path.isdir(fpath):
                    shutil.rmtree(fpath)
                else:
                    os.remove(fpath)
            except Exception as e:
                print(traceback.format_exc())


def make_dataset_cache(skey):
    download_a_sequence(skey)
    timestamp_ns_s = get_timestamp_ns(skey)
    gt_provider = get_gt_provider(skey)
    gen_img_pth(skey=skey, gt_provider=gt_provider, timestamp_ns_s=timestamp_ns_s)
    gen_seg_pth(skey=skey, gt_provider=gt_provider, timestamp_ns_s=timestamp_ns_s)
    gen_gaze_obj_json(skey=skey, gt_provider=gt_provider, timestamp_ns_s=timestamp_ns_s)
    merge_img_seg_gaze_obj_batch(skey=skey)

    del gt_provider
    clean(skey=skey)


if __name__ == '__main__':
    pass
    
    if preset.pc_name == 'XPS':
        make_dataset_cache(skey='Apartment_release_golden_skeleton_seq100_10s_sample_M1292')
    else:
        for skey, info in tqdm(list(sequences.items())[:1]):
            try:
                make_dataset_cache(skey=skey)
            except Exception as e:
                print(traceback.format_exc())

    """
    python e_preprocess_scripts/aria_adt/load_and_make_cache.py
    
    /home/hongyiz/DriverD/b_data_train/data_b_cache
    /home/hongyiz/DynamicFocus
    """
