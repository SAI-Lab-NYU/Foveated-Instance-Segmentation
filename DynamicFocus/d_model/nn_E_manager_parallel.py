import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from d_model.nn_C1_mobilenetv2 import MobileNetV2
from typing import List, Optional
from utility.fctn import save_text

from e_preprocess_scripts.a_preprocess_tools_parallel import CustomDataLoader

from d_model.nn_A4_earlystop import EarlyStopMin, EarlyStopMax

from e_preprocess_scripts.b2_preprocess_lvis import DatasetLVIS
import argparse
import shutil
from pprint import pprint

from d_model.nn_B4_Unet6 import UNet6
from d_model.nn_B7_FovSimModule import FovSimModule
from d_model.nn_B6_SegNet import SegNet
from d_model.nn_D1_seger_zoom import SegerZoom
from d_model.nn_D2_seger_zoomcropcat import SegerZoomCropCat
from d_model.nn_D3_seger_average import SegerAverage
from d_model.nn_D4_seger_uniform import SegerUniform
from d_model.nn_D5_seger_crop import SegerCrop
from d_model.nn_D6_serger_zoom_withembed import SegerZoomEmbed

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

import preset
from d_model.nn_A0_utils import init_weights_random, calc_model_memsize, RAM, try_gpu, init_weights_zero
from d_model.nn_A2_loss import BMSELoss, BCOSIMLoss, WCELoss
from d_model.nn_A3_metrics import evaluate_segmentation, evaluate_classification
from utility.plot_tools import plt_multi_imgshow, plt_show

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = '1234'

    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def delete_subfolders_without_all_required_files(root_folder, required_files):
    # Iterate over all items in the root folder
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)

        # Check if the item is a directory (subfolder)
        if os.path.isdir(subfolder_path):
            # Get a list of all files in the current subfolder
            files_in_subfolder = []
            for root, dirs, files in os.walk(subfolder_path):
                files_in_subfolder.extend(files)
                # No need to go deeper into subfolders
                break

                # Check if all required files are present in the current subfolder
            if all(req_file in files_in_subfolder for req_file in required_files):
                contains_all_required_files = True
            else:
                contains_all_required_files = False

            # If the subfolder doesn't contain all of the required files, delete it
            if not contains_all_required_files:
                print(f"Deleting subfolder: {subfolder_path}")
                shutil.rmtree(subfolder_path)


class ModelManager:

    def __init__(self, model, model_name, device, refresh=False):
        self.device = device

        self.model = model
        self.model_name = model_name

        self.fpath_training_records = os.path.join(preset.dpath_training_records, f"{self.model_name}")
        os.makedirs(self.fpath_training_records, exist_ok=True)

        self.recorder = SummaryWriter(self.fpath_training_records)

        self.fpath_work_pt = os.path.join(self.fpath_training_records, f'params.pt')
        self.fpath_view_pth = os.path.join(self.fpath_training_records, f'params.pth')
        self.fpath_msg_json = os.path.join(self.fpath_training_records, f'msg.json')

        loaded = False

        if not refresh:
            print(f'\nload NN d_model state from {self.fpath_work_pt}')
            loaded = self.load_model(self.fpath_work_pt)
        if not loaded:
            print('\nload NN d_model state fail ; init state')
            self.init_model()

    def init_model(self):
        self.model.module.gen_seg.apply(init_weights_random)
        self.model.module.gen_cls.apply(init_weights_random)

        take_model = self.model
        check_model = take_model
        if isinstance(take_model, nn.DataParallel):
            check_model = take_model.module

        if isinstance(check_model, SegerZoom) or isinstance(check_model, SegerZoomCropCat):
            self.model.gen_seg_0.apply(init_weights_zero)

    def load_model(self, fpath_model):
        loaded = False
        try:
            self.model.load_state_dict(torch.load(fpath_model))
            self.model.eval()
            loaded = True
        except Exception as err:
            # print(traceback.format_exc())
            pass
        return loaded

    def save_model_state(self, fpath_model, msg=''):
        training = self.model.training
        self.model.train(False)

        # back up in case the model is not fully saved
        fpath_bak = f"{fpath_model}.bak"

        if os.path.exists(fpath_model): os.rename(fpath_model, fpath_bak)
        torch.save(self.model.module.state_dict(), fpath_model)
        if os.path.exists(fpath_bak): os.remove(fpath_bak)

        self.model.train(training)

        if msg:
            save_text(msg, fpath_model + '.msg.txt')

        print(f'save to {self.fpath_view_pth}')



    def train(self, take_model: nn.Module, target_device: str, loss_fctn_seg: Optional[nn.modules.loss._Loss], loss_fctn_cls: Optional[nn.modules.loss._Loss], wloss: List, optimizer: torch.optim.Optimizer,
              datasetloader_train: CustomDataLoader,
              batch_size=50,
              lr_scheduler = None):
        optimizer.zero_grad()
        loss_weight_s_train = []

        for i, data in enumerate(datasetloader_train.get_iterator(batch_size=batch_size, device=target_device, shuffle=True)):
        
            x_bx4xHxW, x_bx2, y_bx1xHxW, y_bx1 = data
            b, _, H, W = x_bx4xHxW.shape
            tensor_train_loss = None

            check_model = take_model
            while isinstance(check_model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                check_model = check_model.module
            if isinstance(check_model, SegerZoom):
                y_pred_gs_bx1xHSxWS, grid_pred_bxHSxWSx2, dmap_pred_ds_bx1xHSxWS, x_ds_rgbaf_bxC1xHSxWS, x_gs_rgbaf_bxC1xHSxWS, y_pred_bxK = take_model(x_bx4xHxW, x_bx2)

                y_real_gs_bx1xHSxWS = check_model.downsample_y_gridsp_real_Bx1xHSxWS(y_bx1xHxW, grid_pred_bxHSxWSx2)
                y_real_ds_bx1xHSxWS = check_model.downsample_y_avepool_real_Bx1xHDxWD(y_bx1xHxW)

                w1, w2, w3 = wloss
                tensor_train_loss = (
                        w1 * loss_fctn_seg(dmap_pred_ds_bx1xHSxWS, y_real_ds_bx1xHSxWS) +
                        w2 * loss_fctn_seg(y_pred_gs_bx1xHSxWS, y_real_gs_bx1xHSxWS) +
                        w3 * loss_fctn_cls(y_pred_bxK, y_bx1)
                )


            elif isinstance(check_model, SegerAverage) or isinstance(check_model, SegerUniform):

                y_pred_gs_bx1xHSxWS, x_ds_rgbaf_bxC1xHSxWS, y_pred_bxK = take_model(x_bx4xHxW, x_bx2)
                y_real_ds_bx1xHSxWS = check_model.downsample_y_maxpool_real_Bx1xHSxWS(y_bx1xHxW)
                tensor_train_loss = loss_fctn_seg(y_pred_gs_bx1xHSxWS,
                                                  y_real_ds_bx1xHSxWS)

            cur_train_loss_item = tensor_train_loss.item()
            tensor_train_loss.backward()

            loss_weight_s_train.append([cur_train_loss_item, b])


        final_train_loss_item = sum([ls * wt for ls, wt in loss_weight_s_train]) / sum([wt for ls, wt in loss_weight_s_train])

        if np.isnan(final_train_loss_item):
            raise Exception('shut down for nan loss')

        optimizer.step()
        lr_scheduler.step(final_train_loss_item)
        cur_lr = optimizer.param_groups[0]['lr']

        return cur_lr, final_train_loss_item

    def predict(self, *args):

        mgpu = RAM()

        mgpu.x_bx4xHxW, mgpu.x_bx2 = args
        b, _, H, W = mgpu.x_bx4xHxW.shape
        mgpu.y_pred_bx1xHxW = None

        take_model = self.model
        check_model = take_model
        while isinstance(check_model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            check_model = check_model.module

        check_model.eval()
        take_model.eval()

        with torch.no_grad():
            if isinstance(check_model, SegerZoom):

                mgpu.y_pred_gs_bx1xHSxWS, mgpu.grid_pred_bxHSxWSx2, mgpu.dmap_pred_ds_bx1xHSxWS, mgpu.x_ds_rgbaf_bxC1xHSxWS, mgpu.x_gs_rgbaf_bxC1xHSxWS, mgpu.y_pred_bxK = take_model(mgpu.x_bx4xHxW, mgpu.x_bx2)
                del mgpu.dmap_pred_ds_bx1xHSxWS
                del mgpu.x_ds_rgbaf_bxC1xHSxWS
                del mgpu.x_gs_rgbaf_bxC1xHSxWS

                mgpu.y_pred_bx1xHxW = check_model.output_y_pred_Bx1xHxW(mgpu.y_pred_gs_bx1xHSxWS, mgpu.grid_pred_bxHSxWSx2, mgpu.x_bx4xHxW)

                del mgpu.y_pred_gs_bx1xHSxWS
                del mgpu.grid_pred_bxHSxWSx2
            elif isinstance(check_model, SegerAverage) or isinstance(check_model, SegerUniform):
                mgpu.y_pred_gs_bx1xHSxWS, mgpu.x_ds_rgbaf_bxC1xHSxWS, mgpu.y_pred_bxK = take_model(mgpu.x_bx4xHxW, mgpu.x_bx2)
                del mgpu.x_ds_rgbaf_bxC1xHSxWS
                mgpu.y_pred_bx1xHxW = check_model.output_y_pred_Bx1xHxW(mgpu.y_pred_gs_bx1xHSxWS)
                del mgpu.y_pred_gs_bx1xHSxWS

        # print(self.model)
        return mgpu.y_pred_bx1xHxW, mgpu.y_pred_bxK

    def get_metrics(self, class_num, datasetloader: CustomDataLoader, namekeys, dataset_partition, batch_size=5, show=False, save=False,
                    rank=0, world_size=1,):

        target_device = self.device
        xrange = trange if show else range
        mgpu = RAM()
        namekeys = namekeys[rank::world_size]

        seg_iou_s, seg_f1_s, seg_accuracy_s, seg_precision_s, seg_recall_s = [], [], [], [], []
        y_pred_BxK = []
        y_Bx1 = []

        for i, data in enumerate(datasetloader.get_iterator(batch_size=batch_size, device=target_device, shuffle=False, xrange=xrange)):
            mgpu.x_bx4xHxW, mgpu.x_bx2, mgpu.y_bx1xHxW, mgpu.y_bx1 = data

            mgpu.y_pred_bx1xHxW, mgpu.y_pred_bxK = self.predict(mgpu.x_bx4xHxW, mgpu.x_bx2)

            del mgpu.x_bx4xHxW
            del mgpu.x_bx2

            seg_iou_B, seg_f1_B, seg_accuracy_B, seg_precision_B, seg_recall_B = evaluate_segmentation(mgpu.y_pred_bx1xHxW, mgpu.y_bx1xHxW)

            y_pred_BxK.extend(mgpu.y_pred_bxK.tolist())
            y_Bx1.extend(mgpu.y_bx1.tolist())

            del mgpu.y_pred_bx1xHxW
            del mgpu.y_bx1xHxW

            seg_iou_s.extend(seg_iou_B)
            seg_f1_s.extend(seg_f1_B)
            seg_accuracy_s.extend(seg_accuracy_B)
            seg_precision_s.extend(seg_precision_B)
            seg_recall_s.extend(seg_recall_B)

            mgpu.delete_all()
            # mgpu.show_cuda_info()

        y_pred_BxK = torch.Tensor(y_pred_BxK).to(dtype=torch.float32)
        y_Bx1 = torch.Tensor(y_Bx1).to(dtype=torch.int64)
        f1_per_class, accuracy_per_class, precision_per_class, recall_per_class = evaluate_classification(predict_BxK=y_pred_BxK, target_Bx1=y_Bx1, class_num=class_num)

        # cls_f1_s.extend(cls_f1_B)
        # cls_accuracy_s.extend(cls_accuracy_B)
        # cls_precision_s.extend(cls_precision_B)
        # cls_recall_s.extend(cls_recall_B)
        #
        mgpu.gc()  # since prediction is large so need clean cuda memory every epoch
        # mgpu.show_cuda_info()

        labelkey2bidxs = {}
        labelkey2kidx = {}
        for bidx, namekey in enumerate(namekeys):
            labelkey = namekey.split('_')[0]
            k = int(namekey.split('_')[2][1:])
            labelkey2kidx[labelkey] = k
            if labelkey not in labelkey2bidxs:
                labelkey2bidxs[labelkey] = []
            labelkey2bidxs[labelkey].append(bidx)

        col2elems = {'label': [],
                     'seg_iou': [],
                     'cls_f1': [],
                     'seg_f1': [],
                     'cls_accuracy': [],
                     'seg_accuracy': [],
                     'cls_precision': [],
                     'seg_precision': [],
                     'cls_recall': [],
                     'seg_recall': []
                     }

        for labelkey, bidxs in labelkey2bidxs.items():
            seg_iou = np.mean(np.array(seg_iou_s)[bidxs])
            seg_f1 = np.mean(np.array(seg_f1_s)[bidxs])
            seg_accuracy = np.mean(np.array(seg_accuracy_s)[bidxs])
            seg_precision = np.mean(np.array(seg_precision_s)[bidxs])
            seg_recall = np.mean(np.array(seg_recall_s)[bidxs])

            cls_f1 = f1_per_class[labelkey2kidx[labelkey]]
            cls_accuracy = accuracy_per_class[labelkey2kidx[labelkey]]
            cls_precision = precision_per_class[labelkey2kidx[labelkey]]
            cls_recall = recall_per_class[labelkey2kidx[labelkey]]

            col2elems['label'].append(labelkey)
            col2elems['seg_iou'].append(seg_iou)
            col2elems['seg_f1'].append(seg_f1)
            col2elems['seg_accuracy'].append(seg_accuracy)
            col2elems['seg_precision'].append(seg_precision)
            col2elems['seg_recall'].append(seg_recall)

            col2elems['cls_f1'].append(cls_f1)
            col2elems['cls_accuracy'].append(cls_accuracy)
            col2elems['cls_precision'].append(cls_precision)
            col2elems['cls_recall'].append(cls_recall)

        mean_seg_iou = np.array(col2elems['seg_iou']).mean()
        mean_seg_f1 = np.array(col2elems['seg_f1']).mean()
        mean_seg_accuracy = np.array(col2elems['seg_accuracy']).mean()
        mean_seg_precision = np.array(col2elems['seg_precision']).mean()
        mean_seg_recall = np.array(col2elems['seg_recall']).mean()
        mean_cls_f1 = np.array(col2elems['cls_f1']).mean()
        mean_cls_accuracy = np.array(col2elems['cls_accuracy']).mean()
        mean_cls_precision = np.array(col2elems['cls_precision']).mean()
        mean_cls_recall = np.array(col2elems['cls_recall']).mean()

        col2elems['label'].append('MEAN')
        col2elems['seg_iou'].append(mean_seg_iou)
        col2elems['seg_f1'].append(mean_seg_f1)
        col2elems['seg_accuracy'].append(mean_seg_accuracy)
        col2elems['seg_precision'].append(mean_seg_precision)
        col2elems['seg_recall'].append(mean_seg_recall)
        col2elems['cls_f1'].append(mean_cls_f1)
        col2elems['cls_accuracy'].append(mean_cls_accuracy)
        col2elems['cls_precision'].append(mean_cls_precision)
        col2elems['cls_recall'].append(mean_cls_recall)

        if save:
            df = pd.DataFrame(col2elems)

            fpath_metrics = os.path.join(preset.dpath_training_records, f"{self.model_name}", f'{dataset_partition}.metrics.csv')

            df.to_csv(fpath_metrics, index=False)
            print(f"SAVE metrics to {fpath_metrics}")

        mgpu.delete_all()
        mgpu.gc()
        return mean_seg_iou, mean_seg_f1, mean_seg_accuracy, mean_seg_precision, mean_seg_recall, mean_cls_f1, mean_cls_accuracy, mean_cls_precision, mean_cls_recall

    def plot_figure(self, datasetloader: CustomDataLoader, fname=f"plot_figure.png"):
        is_training_model = self.model.training
        self.model.eval()
        mgpu = RAM()

        self.model.eval()

        with torch.no_grad():
            if isinstance(self.model, SegerZoom):
                rows = 5
                mgpu.x_bx4xHxW, mgpu.x_bx2, mgpu.y_bx1xHxW, mgpu.y_bx1 = next(iter(datasetloader.get_iterator(batch_size=rows, device=target_device, shuffle=True)))

                B, _, H, W = mgpu.x_bx4xHxW.shape
                rows = B

                mgpu.y_real_ds_bx1xHSxWS = self.model.downsample_y_avepool_real_Bx1xHDxWD(mgpu.y_bx1xHxW)

                mgpu.y_pred_gs_bx1xHSxWS, mgpu.grid_pred_bxHSxWSx2, mgpu.dmap_pred_ds_bx1xHSxWS, mgpu.x_ds_rgbaf_bxC1xHSxWS, mgpu.x_gs_rgbaf_bxC1xHSxWS, mgpu.y_pred_bxK = self.model(mgpu.x_bx4xHxW, mgpu.x_bx2)

                # plt_multi_imgshow([mgpu.x_ds_rgbaf_bxC1xHSxWS[i, :-1] for i in range(5)], row_col=(1, 5))
                # plt_show()
                #
                # plt_multi_imgshow([mgpu.x_ds_rgbaf_bxC1xHSxWS[i, -1:] for i in range(5)], row_col=(1, 5))
                # plt_show()
                # plt_multi_imgshow([mgpu.x_gs_rgbaf_bxC1xHSxWS[i, :-1] for i in range(5)], row_col=(1, 5))
                # plt_show()
                # plt_multi_imgshow([mgpu.x_gs_rgbaf_bxC1xHSxWS[i, -1:] for i in range(5)], row_col=(1, 5))
                # plt_show()

                mgpu.y_real_gs_bx1xHSxWS = self.model.downsample_y_gridsp_real_Bx1xHSxWS(mgpu.y_bx1xHxW, mgpu.grid_pred_bxHSxWSx2)

                mgpu.y_pred_bx1xHxW = self.model.output_y_pred_Bx1xHxW(mgpu.y_pred_gs_bx1xHSxWS, mgpu.grid_pred_bxHSxWSx2, mgpu.x_bx4xHxW)

                # print('beg plot')
                # pprint(mgpu.y_pred_bxK)
                # pprint(torch.argmax(mgpu.y_pred_bxK, dim=1))
                # print('end plot')

                imgs = []
                titles = []

                for b in range(B):
                    imgs.extend([

                        mgpu.x_bx4xHxW[b, :4],
                        torch.cat([mgpu.x_ds_rgbaf_bxC1xHSxWS[b, :3], mgpu.dmap_pred_ds_bx1xHSxWS[b]]),
                        torch.cat([mgpu.x_ds_rgbaf_bxC1xHSxWS[b, :3], mgpu.y_real_ds_bx1xHSxWS[b]]),

                        torch.cat([mgpu.x_gs_rgbaf_bxC1xHSxWS[b, :3], mgpu.y_pred_gs_bx1xHSxWS[b]]),
                        torch.cat([mgpu.x_gs_rgbaf_bxC1xHSxWS[b, :3], mgpu.y_real_gs_bx1xHSxWS[b]]),
                        mgpu.y_pred_bx1xHxW[b],
                        mgpu.y_bx1xHxW[b]

                    ])
                    titles.extend([
                        f"orig_rgba_4xHxW_{b}",
                        f"dmap_pred_aver_rgba_C1xHSxWS_{b}",
                        f"dmap_real_aver_rgba_C1xHSxWS_{b}",
                        f"sgmt_pred_grid_rgba_C1xHSxWS_{b}",
                        f"sgmt_real_grid_rgba_C1xHSxWS_{b}",

                        f"fina_pred_back_rgba_1xHxW_{b}",
                        f"fina_real_back_rgba_1xHxW_{b}"
                    ])

                plt_multi_imgshow(imgs, titles, row_col=(B, 7))
                fpath_plot = os.path.join(self.fpath_training_records, fname)
                plt.savefig(fpath_plot)
                plt.close('all')
                print(f"save to {fpath_plot}")

        mgpu.delete_all()
        mgpu.gc()
        self.model.train(is_training_model)


def get_sys_kwargs():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Preprocess the Cityscape dataset with specified parameters.")

    cbase_seg0 = 4
    nlayer_seg0 = 3
    cbase_seg1 = 16
    nlayer_seg1 = 4

    if preset.pc_name == preset.PC_NAME_server_H100 or preset.pc_name == preset.PC_NAME_server_4090:
        cbase_seg0 = 8
        nlayer_seg0 = 3
        cbase_seg1 = 64
        nlayer_seg1 = 5

    parser.add_argument('--Kblur', type=int, default=40, required=False, help=" (e.g., 2 4 8 ,16,32).")
    parser.add_argument('--Kprio', type=int, default=40, required=False, help=" (e.g., 2 4 8 ,16,32).")
    parser.add_argument('--Kgrid', type=int, default=40, required=False, help=" (e.g., 2 4 8 ,16,32).")

    parser.add_argument('--downsample_factor', type=int, default=4, required=False, help="downsample factor")
    parser.add_argument('--downsample_factor_deformation', type=int, default=2, required=False, help="downsample factor deformation")

    parser.add_argument('--cbase_seg0', type=int, default=cbase_seg0, required=False, help="Base channel count for segment 0.")
    parser.add_argument('--cbase_seg1', type=int, default=cbase_seg1, required=False, help="Base channel count for segment 1.")
    parser.add_argument('--nlayer_seg0', type=int, default=nlayer_seg0, required=False, help="Number of layers for segment 0.")
    parser.add_argument('--nlayer_seg1', type=int, default=nlayer_seg1, required=False, help="Number of layers for segment 1.")
    parser.add_argument('--param_priori', action='store_true', help="Enable priori")
    parser.add_argument('--param_square_focus', action='store_true', help="Enable square_focus")

    parser.add_argument('--dataset_marker_train', type=str, default="", required=False, help="marker folder of dataset")
    parser.add_argument('--dataset_marker_valid', type=str, default="", required=False, help="marker folder of dataset")
    parser.add_argument('--batch_size_train', type=int, default=25, required=False, help="batch size for training")
    parser.add_argument('--batch_size_valid', type=int, default=10, required=False, help="batch size for validation")

    parser.add_argument('--Module_Loss', type=str, default='SegerZoom_DBMSELoss', required=False, help="One or more Module_Loss to use [Module]_[Loss]")
    parser.add_argument('--modelname', type=str, default="", required=False, help="Name of the model.")
    parser.add_argument('--dataset_class', type=str, default="", required=False, help="dataset name")

    # 添加 train 和 metrics 参数，作为独立的布尔选项
    parser.add_argument('--train', action='store_true', help="Enable training mode.")
    parser.add_argument('--metrics', action='store_true', help="Enable metrics mode.")
    parser.add_argument('--clean_records', action='store_true', help='delete subfolders without all required files')

    kwargs = parser.parse_args()

    return kwargs

def main(rank, world_size):
    ddp_setup(rank, world_size)

    dp_train = 'train'
    dp_valid = 'valid'

    in_channels = 4
    out_channels = 1
    class_num = 5
    wloss = [0.5, 0.5, 1]

    base_module_deformation = UNet6
    base_module = UNet6
    classify_module = MobileNetV2

    Skwargs = get_sys_kwargs()

    if Skwargs.clean_records:
        delete_subfolders_without_all_required_files(preset.dpath_training_records, ['train.metrics.csv', 'valid.metrics.csv'])

    batch_size_train = Skwargs.batch_size_train
    batch_size_val = Skwargs.batch_size_valid

    pprint(Skwargs._get_kwargs())

    # target_device = try_gpu(gpu_index=Skwargs.gpu_idxs[0])

    target_device = rank
    torch.cuda.set_device(rank)


    if not Skwargs.dataset_class:
        Skwargs.dataset_class = 'DatasetLVIS'

    datasetloader_train = None
    datasetloader_valid = None

    from torch.utils.data import DistributedSampler
    batch_size_train = batch_size_train//world_size
    batch_size_val = batch_size_val//world_size
    print(batch_size_train)

    if Skwargs.dataset_class == 'DatasetLVIS':
        # 创建 DistributedSampler
        train_sampler = DistributedSampler(DatasetLVIS(marker=Skwargs.dataset_marker_train, dataset_partition=dp_train), num_replicas=world_size, rank=rank)
        valid_sampler = DistributedSampler(DatasetLVIS(marker=Skwargs.dataset_marker_valid, dataset_partition=dp_valid), num_replicas=world_size, rank=rank)


        datasetloader_train = CustomDataLoader(DatasetLVIS(marker=Skwargs.dataset_marker_train, dataset_partition=dp_train), sampler=train_sampler, cache=True, xrange=trange)
        datasetloader_valid = CustomDataLoader(DatasetLVIS(marker=Skwargs.dataset_marker_valid, dataset_partition=dp_valid), sampler=valid_sampler, cache=True, xrange=trange)


    module_name, lossname = Skwargs.Module_Loss.split('_')

    nn_loss_fctn_seg = BMSELoss()
    nn_loss_fctn_cls = WCELoss(class_num=class_num)

    nn_module = None
    if module_name == 'SegerZoom':
        nn_module = SegerZoom(base_module_deformation=base_module_deformation,
                              base_module=base_module,
                              classify_module=classify_module,
                              in_channels=in_channels,
                              out_channels=out_channels,
                              class_num=class_num,
                              downsample_factor=Skwargs.downsample_factor,
                              downsample_factor_deformation=Skwargs.downsample_factor_deformation,
                              kernel_gridsp=Skwargs.Kblur + 1,
                              kernel_gblur=Skwargs.Kgrid + 1,
                              cbase_seg0=Skwargs.cbase_seg0,
                              cbase_seg=Skwargs.cbase_seg1,
                              nlayer_seg0=Skwargs.nlayer_seg0,
                              nlayer_seg=Skwargs.nlayer_seg1,
                              priori=Skwargs.param_priori,
                              kernel_priori=Skwargs.Kprio + 1,
                              square_focus=Skwargs.param_square_focus,
                              )
        calc_model_memsize(nn_module.gen_seg_0, label='gen_seg_0')

    elif module_name == 'SegerAverage':
        nn_module = SegerAverage(base_module=base_module, in_channels=in_channels,
                                 out_channels=out_channels,
                                 downsample_factor=Skwargs.downsample_factor,
                                 cbase_seg=Skwargs.cbase_seg1,
                                 nlayer_seg=Skwargs.nlayer_seg1)
    elif module_name == 'SegerUniform':
        nn_module = SegerUniform(base_module=base_module, in_channels=in_channels,
                                 out_channels=out_channels,
                                 downsample_factor=Skwargs.downsample_factor,
                                 cbase_seg=Skwargs.cbase_seg1,
                                 nlayer_seg=Skwargs.nlayer_seg1)

    calc_model_memsize(nn_module.gen_seg, label='gen_seg')
    calc_model_memsize(nn_module.gen_cls, label='gen_cls')

    calc_model_memsize(nn_module, label='enire_model')

    modelname = Skwargs.modelname

    if not modelname:
        modelname = ""
        modelname += datetime.now().strftime('D%y%m%d_T%H%M%S')
        modelname += f"_{datasetloader_train.dataset.__class__.__name__.replace('Dataset', '')}"
        modelname += f"_{len(datasetloader_train.dataset)}x{datasetloader_train.dataset.HC}x{datasetloader_train.dataset.WC}"
        modelname += f"_{datasetloader_train.dataset.HC // Skwargs.downsample_factor}x{datasetloader_train.dataset.WC // Skwargs.downsample_factor}"
        modelname += f"_{datasetloader_train.dataset.HC // Skwargs.downsample_factor // Skwargs.downsample_factor_deformation}x{datasetloader_train.dataset.WC // Skwargs.downsample_factor // Skwargs.downsample_factor_deformation}"

        if module_name in ['SegerZoom', 'SegerZoomCropCat']:
            modelname += f"_{base_module_deformation.__name__}"

        modelname += f"_{base_module.__name__}"

        modelname += f"_{nn_module.__class__.__name__}"

        modelname += f"_{nn_loss_fctn_seg.__class__.__name__.replace('Loss', '')}"

        if module_name in ['SegerZoom', 'SegerZoomCropCat']:
            modelname += f"_cbase-{Skwargs.cbase_seg0}-{Skwargs.cbase_seg1}"
            modelname += f"_nlayer-{Skwargs.nlayer_seg0}-{Skwargs.nlayer_seg1}"
            modelname += f"_kblur{Skwargs.Kblur}"
            modelname += f"_kgrid{Skwargs.Kgrid}"
            modelname += f"_kprio{Skwargs.Kprio}"

            modelname += f"_wloss-{''.join([str(w) for w in wloss])}"
            modelname += f"_pri-{int(Skwargs.param_priori)}"
            modelname += f"_sqf-{int(Skwargs.param_square_focus)}"
        else:
            modelname += f"_cbase-{Skwargs.cbase_seg1}"
            modelname += f"_nlayer-{Skwargs.nlayer_seg1}"

    print(modelname)

    print(target_device)

    n_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {n_gpus}")
    torch.cuda.set_device(rank)  
    nn_module = nn_module.to(rank) 
    nn_module = DDP(nn_module, device_ids=[rank], output_device=rank)

    # if target_device == 'cuda':
    #     nn_module = nn.DataParallel(nn_module)
    #     cudnn.benchmark = True

    mm = ModelManager(nn_module, modelname, target_device, refresh=False)

    epoch = 500
    save_per_N_epoch = 10
    plot_per_N_epoch = 50
    eval_per_N_epoch = 50

    if preset.pc_name == preset.PC_NAME_server_H100:
        epoch = 500
        save_per_N_epoch = 50
        plot_per_N_epoch = 50
        eval_per_N_epoch = 100

    print_per_N_epoch = 10

    init_lr = 0.01
    decay = 0.95

    optimizer = torch.optim.NAdam(mm.model.parameters(), lr=init_lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=decay, patience=5)

    earlystop = EarlyStopMax()
    save_trigger = False
    mgpu = RAM()

    if Skwargs.train:

        for ep in trange(epoch):
            train_sampler.set_epoch(ep)
            mm.model.train(True)
            cur_lr, cur_loss_train = mm.train(nn_module, target_device, loss_fctn_seg=nn_loss_fctn_seg, loss_fctn_cls=nn_loss_fctn_cls, wloss=wloss, optimizer=optimizer, datasetloader_train=datasetloader_train, batch_size=batch_size_train, lr_scheduler=lr_scheduler)

            mm.model.train(False)
            # mgpu.show_cuda_info()
            save_trigger = False

            lr_scheduler.step(cur_loss_train)
            mm.recorder.add_scalar('train_loss', cur_loss_train, ep)

            if ep % save_per_N_epoch == 0:
                print(ep, save_per_N_epoch)
                save_trigger = True

            if ep % print_per_N_epoch == 0:
                print(f"ep={ep} loss={cur_loss_train:.6f} lr={cur_lr:.6f} trig={int(save_trigger)}")

            if ep % plot_per_N_epoch == 0:
                mm.plot_figure(datasetloader_train, fname=f"ep{ep}.train.plot.png")
                mm.plot_figure(datasetloader_valid, fname=f"ep{ep}.valid.plot.png")

            if save_trigger:
                res_bind = mm.get_metrics(class_num=class_num, datasetloader=datasetloader_valid,
                                        namekeys=datasetloader_valid.dataset.get_namekeys(),
                                        dataset_partition=datasetloader_valid.dataset.dataset_partition,
                                        batch_size=batch_size_val, show=False, save=False, rank=rank, world_size=world_size)

                mean_seg_iou, mean_seg_f1, mean_seg_accuracy, mean_seg_precision, mean_seg_recall, mean_cls_f1, mean_cls_accuracy, mean_cls_precision, mean_cls_recall = res_bind
                mean_score = 0.5 * mean_seg_f1 + 0.5 * mean_cls_f1
                if earlystop.check(mean_seg_iou):
                    if rank==0:
                        mm.save_model_state(mm.fpath_work_pt, msg=f'Best Epoch : {ep} loss={cur_loss_train:.6f}')
                        mm.get_metrics(class_num=class_num, datasetloader=datasetloader_train, namekeys=datasetloader_train.dataset.get_namekeys(), dataset_partition=datasetloader_train.dataset.dataset_partition, batch_size=batch_size_val,
                                    show=True, save=True, rank=rank, world_size=world_size)
                        mm.get_metrics(class_num=class_num, datasetloader=datasetloader_valid, namekeys=datasetloader_valid.dataset.get_namekeys(), dataset_partition=datasetloader_valid.dataset.dataset_partition, batch_size=batch_size_val,
                                show=True, save=True, rank=rank, world_size=world_size)
            torch.distributed.barrier()

            mgpu = RAM()
            mgpu.delete_all()
            mgpu.gc()

        mm.recorder.flush()
        mm.recorder.close()

    # if Skwargs.metrics:
    #     mm.load_model(fpath_model=mm.fpath_work_pt)

    #     mm.model.train(False)
    #     mm.plot_figure(datasetloader_train, fname=f"final.train.plot.png")
    #     mm.plot_figure(datasetloader_valid, fname=f"final.valid.plot.png")
    #     mm.get_metrics(class_num=class_num, datasetloader=datasetloader_train, namekeys=datasetloader_train.dataset.get_namekeys(), dataset_partition=datasetloader_train.dataset.dataset_partition, batch_size=batch_size_val, show=True,
    #                    save=True)
    #     mm.get_metrics(class_num=class_num, datasetloader=datasetloader_valid, namekeys=datasetloader_valid.dataset.get_namekeys(), dataset_partition=datasetloader_valid.dataset.dataset_partition, batch_size=batch_size_val, show=True,
    #                    save=True)

    destroy_process_group()


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, ), nprocs=world_size)
"""
TODO add GPU manually assign

# on laptop
python d_model/nn_E_manager.py --train --metrics --Module_Loss SegerAverage_BMSELoss --dataset_marker_train sp10 --dataset_marker_valid sp10
python d_model/nn_E_manager.py --train --metrics --Module_Loss SegerUniform_BMSELoss --dataset_marker_train sp10 --dataset_marker_valid sp10 
python d_model/nn_E_manager.py --train --metrics --Module_Loss SegerZoom_BMSELoss --dataset_marker_train sp10 --dataset_marker_valid sp10 --param_priori --param_square_focus
python d_model/nn_E_manager.py --train --metrics --Module_Loss SegerZoom_BMSELoss --dataset_marker_train sp10 --dataset_marker_valid sp10 --param_priori 
python d_model/nn_E_manager.py --train --metrics --Module_Loss SegerZoom_BMSELoss --dataset_marker_train sp10 --dataset_marker_valid sp10 --param_square_focus
python d_model/nn_E_manager.py --train --metrics --Module_Loss SegerZoom_BMSELoss --dataset_marker_train sp10 --dataset_marker_valid sp10



# on server


CUDA_VISIBLE_DEVICES=3 python d_model/nn_E_manager.py --train --metrics --Module_Loss SegerZoom_BMSELoss --dataset_marker_train sp500 --dataset_marker_valid sp100


git pull; tmux kill-server ; rm -r a_records_train;

git pull; tmux kill-server ; rm -r a_records_train ; \
tmux new-session -d "CUDA_VISIBLE_DEVICES=0 python d_model/nn_E_manager.py --train --metrics --Module_Loss SegerAverage_BMSELoss --dataset_marker_train sp100 --dataset_marker_valid sp100 ; bash" && \
tmux new-session -d "CUDA_VISIBLE_DEVICES=1 python d_model/nn_E_manager.py --train --metrics --Module_Loss SegerUniform_BMSELoss --dataset_marker_train sp100 --dataset_marker_valid sp100 ; bash" && \
tmux new-session -d "CUDA_VISIBLE_DEVICES=0 python d_model/nn_E_manager.py --train --metrics --Module_Loss SegerZoom_BMSELoss --dataset_marker_train sp100 --dataset_marker_valid sp100 ; bash" && \
tmux new-session -d "CUDA_VISIBLE_DEVICES=1 python d_model/nn_E_manager.py --train --metrics --Module_Loss SegerZoom_BMSELoss --dataset_marker_train sp100 --dataset_marker_valid sp100 --param_square_focus ; bash" && \
tmux new-session -d "CUDA_VISIBLE_DEVICES=2 python d_model/nn_E_manager.py --train --metrics --Module_Loss SegerZoom_BMSELoss --dataset_marker_train sp100 --dataset_marker_valid sp100 --param_priori ; bash" && \
tmux new-session -d "CUDA_VISIBLE_DEVICES=3 python d_model/nn_E_manager.py --train --metrics --Module_Loss SegerZoom_BMSELoss --dataset_marker_train sp100 --dataset_marker_valid sp100 --param_priori --param_square_focus ; bash " 

tmux kill-server ; 


tmux new-session -d "CUDA_VISIBLE_DEVICES=0 python d_model/nn_E_manager.py --train --metrics --Module_Loss SegerZoom_BMSELoss --dataset_marker_train sp100 --dataset_marker_valid sp100 --Kgrid 80 --downsample_factor_deformation 1 ; bash" && \
tmux new-session -d "CUDA_VISIBLE_DEVICES=1 python d_model/nn_E_manager.py --train --metrics --Module_Loss SegerZoom_BMSELoss --dataset_marker_train sp100 --dataset_marker_valid sp100 --Kgrid 40 --downsample_factor_deformation 1 ; bash" && \
tmux new-session -d "CUDA_VISIBLE_DEVICES=2 python d_model/nn_E_manager.py --train --metrics --Module_Loss SegerZoom_BMSELoss --dataset_marker_train sp100 --dataset_marker_valid sp100 --Kgrid 20 --downsample_factor_deformation 1 ; bash" && \
tmux new-session -d "CUDA_VISIBLE_DEVICES=3 python d_model/nn_E_manager.py --train --metrics --Module_Loss SegerZoom_BMSELoss --dataset_marker_train sp100 --dataset_marker_valid sp100 --Kgrid 10 --downsample_factor_deformation 1 ; bash" && \

tmux new-session -d "CUDA_VISIBLE_DEVICES=0 python d_model/nn_E_manager.py --train --metrics --Module_Loss SegerZoom_BMSELoss --dataset_marker_train sp100 --dataset_marker_valid sp100 --Kgrid 80 --downsample_factor_deformation 1 ; bash" && \
tmux new-session -d "CUDA_VISIBLE_DEVICES=1 python d_model/nn_E_manager.py --train --metrics --Module_Loss SegerZoom_BMSELoss --dataset_marker_train sp100 --dataset_marker_valid sp100 --Kgrid 40 --downsample_factor_deformation 1 ; bash" && \
tmux new-session -d "CUDA_VISIBLE_DEVICES=2 python d_model/nn_E_manager.py --train --metrics --Module_Loss SegerZoom_BMSELoss --dataset_marker_train sp100 --dataset_marker_valid sp100 --Kgrid 20 --downsample_factor_deformation 1 ; bash" && \
tmux new-session -d "CUDA_VISIBLE_DEVICES=3 python d_model/nn_E_manager.py --train --metrics --Module_Loss SegerZoom_BMSELoss --dataset_marker_train sp100 --dataset_marker_valid sp100 --Kgrid 10 --downsample_factor_deformation 1 ; bash" && \



python d_model/nn_E_manager.py --clean_records

"""
