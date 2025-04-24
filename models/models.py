import torch
import random
import torch.nn as nn
from torch.nn import functional as F
import torchvision
import torchvision.utils as vutils
import torchsnooper
from . import resnet, resnext, mobilenet, hrnetv2_nodownsp, segformer, deeplab
from lib.nn import SynchronizedBatchNorm2d
from dataset import imresize, b_imresize
from builtins import any as b_any

from lib.utils import as_numpy
from utils import colorEncode
from scipy.io import loadmat
import numpy as np
from PIL import Image
from PIL import ImageFilter
import time
import os
import shutil
from scipy import ndimage
import scipy.interpolate
import cv2
import torchvision.models as models
from saliency_network import saliency_network_resnet18, fov_simple, saliency_network_resnet18_stride1
from models.model_utils import Resnet, ResnetDilated, MobileNetV2Dilated, C1DeepSup, C1, PPM, PPMDeepsup, UPerNet
from DynamicFocus.utility.torch_tools import gen_grid_mtx_2xHxW
BatchNorm2d = SynchronizedBatchNorm2d
from pytorch_toolbelt.losses.dice import DiceLoss

from torch.autograd import Variable
import matplotlib.pyplot as plt

import pdb

def generate_colormap_colors(num_colors):
    cmap = plt.get_cmap('hsv')  # You can try 'viridis', 'plasma', 'tab20', etc.
    colors = [cmap(i / num_colors) for i in range(num_colors)]
    # Convert RGBA (0-1) to RGB (0-255)
    colors = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b, _ in colors]
    colors[-1] = (0, 0, 0)
    return colors

def rgb_img(batch_0):
    color_palette = generate_colormap_colors(51)
    batch_0_numpy = batch_0.cpu().numpy()

    # Create an RGB image
    h, w = batch_0_numpy.shape
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
    # Map each class value (0-19) to its corresponding color
    for class_id, color in enumerate(color_palette):
        rgb_image[batch_0_numpy == class_id] = color
    # Step 4: Save the image
    output_image = Image.fromarray(rgb_image)
    return output_image

class SoftDiceLossV1(nn.Module):
    '''
    soft-dice loss, useful in binary segmentation
    '''
    def __init__(self,
                 p=2,
                 smooth=0):
        super(SoftDiceLossV1, self).__init__()
        self.p = p
        self.smooth = smooth

    def forward(self, logits, labels):
        '''
        inputs:
            logits: tensor of shape (N, H, W, ...)
            label: tensor of shape(N, H, W, ...)
        output:
            loss: tensor of shape(1, )
        '''
        logits = torch.permute(logits, (0,2,3,1)).cuda()

        probs = torch.sigmoid(logits)
        numer = (probs * labels).sum()
        denor = (probs.pow(self.p) + labels.pow(self.p)).sum()
        loss = 1. - (2 * numer + self.smooth) / (denor + self.smooth)
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)
        if torch.isnan(input).any():
            print("input contains NaN values!")
        if torch.isinf(input).any():
            print("input contains inf values!")
        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        assert not torch.isnan(loss).any(), "loss contains NaN values!"
        if self.size_average: return loss.mean()
        else: return loss.sum()

class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()
    
    def forward(self, y_pred_ds_Bx1xHSxWS):
        batch_size = y_pred_ds_Bx1xHSxWS.size(0)
        h = y_pred_ds_Bx1xHSxWS.size(2)
        w = y_pred_ds_Bx1xHSxWS.size(3)

        h_tv = torch.abs(y_pred_ds_Bx1xHSxWS[:, :, 1:, :] - y_pred_ds_Bx1xHSxWS[:, :, :-1, :]).sum()
        w_tv = torch.abs(y_pred_ds_Bx1xHSxWS[:, :, :, 1:] - y_pred_ds_Bx1xHSxWS[:, :, :, :-1]).sum()

        count_h = (y_pred_ds_Bx1xHSxWS.size(2) - 1) * y_pred_ds_Bx1xHSxWS.size(3)
        count_w = y_pred_ds_Bx1xHSxWS.size(2) * (y_pred_ds_Bx1xHSxWS.size(3) - 1)
        total_variation_loss = (h_tv / count_h + w_tv / count_w) / batch_size

        return total_variation_loss   

def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

def fillMissingValues_tensor(target_for_interp, copy=False, interp_mode='tri'):
    """
    fill missing values in a tenor

    input shape: [num_classes, h, w]
    output shape: [num_classes, h, w]
    """

    if copy:
        target_for_interp = target_for_interp.clone()

    def getPixelsForInterp(img):
        """
        Calculates a mask of pixels neighboring invalid values -
           to use for interpolation.

        input shape: [num_classes, h, w]
        output shape: [num_classes, h, w]
        """

        invalid_mask = torch.isnan(img)
        kernel = torch.tensor(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), device=invalid_mask.device).unsqueeze(0).unsqueeze(0).expand(1,invalid_mask.shape[0],3,3).float()

        #dilate to mark borders around invalid regions
        if max(invalid_mask.shape) > 512:
            dr = max(invalid_mask.shape)/512
            input = invalid_mask.float().unsqueeze(0)
            shape_ori = (invalid_mask.shape[-2], int(invalid_mask.shape[-1]))
            shape_scaled = (int(invalid_mask.shape[-2]/dr), int(invalid_mask.shape[-1]/dr))
            input_scaled = F.interpolate(input, shape_scaled, mode='nearest').squeeze(0)
            invalid_mask_scaled = input_scaled.unsqueeze(0) # b,c,w,h

            dilated_mask_scaled = torch.clamp(F.conv2d(invalid_mask_scaled, kernel, padding=(1, 1)), 0, 1)
            dilated_mask_scaled_t = dilated_mask_scaled.float()
            dilated_mask = F.interpolate(dilated_mask_scaled_t, shape_ori, mode='nearest').squeeze(0)
        else:

            dilated_mask = torch.clamp(F.conv2d(invalid_mask.float().unsqueeze(0),
                                                kernel, padding=(1, 1)), 0, 1).squeeze(0)

        # pixelwise "and" with valid pixel mask (~invalid_mask)
        masked_for_interp = dilated_mask *  (~invalid_mask).float()
        # Add 4 zeros corner points required for interp2d
        masked_for_interp[:,0,0] *= 0
        masked_for_interp[:,0,-1] *= 0
        masked_for_interp[:,-1,0] *= 0
        masked_for_interp[:,-1,-1] *= 0
        masked_for_interp[:,0,0] += 1
        masked_for_interp[:,0,-1] += 1
        masked_for_interp[:,-1,0] += 1
        masked_for_interp[:,-1,-1] += 1

        return masked_for_interp.bool(), invalid_mask

    def getPixelsForInterp_NB(img):
        """
        Calculates a mask of pixels neighboring invalid values -
           to use for interpolation.
        """
        # mask invalid pixels
        img = img.cpu().numpy()
        invalid_mask = np.isnan(img)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        #dilate to mark borders around invalid regions
        if max(invalid_mask.shape) > 512:
            dr = max(invalid_mask.shape)/512
            input = torch.tensor(invalid_mask.astype('float')).unsqueeze(0)
            shape_ori = (invalid_mask.shape[-2], int(invalid_mask.shape[-1]))
            shape_scaled = (int(invalid_mask.shape[-2]/dr), int(invalid_mask.shape[-1]/dr))
            input_scaled = F.interpolate(input, shape_scaled, mode='nearest').squeeze(0)
            invalid_mask_scaled = np.array(input_scaled).astype('bool')
            dilated_mask_scaled = cv2.dilate(invalid_mask_scaled.astype('uint8'), kernel,
                              borderType=cv2.BORDER_CONSTANT, borderValue=int(0))
            dilated_mask_scaled_t = torch.tensor(dilated_mask_scaled.astype('float')).unsqueeze(0)
            dilated_mask = F.interpolate(dilated_mask_scaled_t, shape_ori, mode='nearest').squeeze(0)
            dilated_mask = np.array(dilated_mask).astype('uint8')
        else:
            dilated_mask = cv2.dilate(invalid_mask.astype('uint8'), kernel,
                              borderType=cv2.BORDER_CONSTANT, borderValue=int(0))

        # pixelwise "and" with valid pixel mask (~invalid_mask)
        masked_for_interp = dilated_mask *  ~invalid_mask
        return masked_for_interp.astype('bool'), invalid_mask

    # Mask pixels for interpolation
    if interp_mode == 'nearest':
        interpolator=scipy.interpolate.NearestNDInterpolator
        mask_for_interp, invalid_mask = getPixelsForInterp_NB(target_for_interp)
    elif interp_mode == 'BI':
        interpolator=scipy.interpolate.LinearNDInterpolator
        mask_for_interp, invalid_mask = getPixelsForInterp_NB(target_for_interp)
    else:
        interpolator=Interp2D(target_for_interp.shape[-2], target_for_interp.shape[-1])
        mask_for_interp, invalid_mask = getPixelsForInterp(target_for_interp)
        if invalid_mask.float().sum() == 0:
            return target_for_interp

    if interp_mode == 'nearest' or interp_mode == 'BI':
        #print('1111111111111111')
        points = np.argwhere(mask_for_interp)  #np array
        #values = target_for_interp[mask_for_interp] #tensor
        values = target_for_interp[mask_for_interp].cpu().numpy() #np
        #print(type(points), type(values))
    else:
        print('222222222222222222')
        points = torch.where(mask_for_interp[0]) # tuple of 2 for (h, w) indices
        points = torch.cat([t.unsqueeze(0) for t in points]) # [2, number_of_points]
        points = points.permute(1,0) # shape: [number_of_points, 2]
        values = target_for_interp.clone()[mask_for_interp].view(mask_for_interp.shape[0],-1).permute(1,0) # shape: [number_of_points, num_classes]
    interp = interpolator(points, values) # return [num_classes, h, w]
    if interp_mode == 'nearest' or interp_mode == 'BI':
        #print('227')
        target_for_interp[invalid_mask] = torch.tensor(interp(np.argwhere(np.array(invalid_mask)))).float().cuda()
        #print('227 done')
    else:
        if not (interp.shape == target_for_interp.shape == invalid_mask.shape and interp.device == target_for_interp.device == invalid_mask.device):
            print('SHAPE: interp={}; target_for_interp={}; invalid_mask={}\n'.format(interp.shape, target_for_interp.shape, invalid_mask.shape))
            print('DEVICE: interp={}; target_for_interp={}; invalid_mask={}\n'.format(interp.device, target_for_interp.device, invalid_mask.device))
        try:
            target_for_interp[invalid_mask] = interp[torch.where(invalid_mask)].clone()
        except:
            print('interp: {}\n'.format(interp))
            print('invalid_mask: {}\n'.format(invalid_mask))
            print('target_for_interp: {}\n'.format(target_for_interp))
        else:
            pass
    return target_for_interp

def create_map(B, hidx_B, widx_B, height=80, width=80, radius=25, max_value=0.5, min_value=0.05):
    """
    Creates a map with shape (B, 1, height, width). Each batch has a point at (hidx, widx)
    with value max_value, decreasing in a cosine style to min_value within a radius.

    Parameters:
        B (int): Batch size.
        hidx_B (torch.Tensor): Tensor of shape (B, 1), y-coordinates.
        widx_B (torch.Tensor): Tensor of shape (B, 1), x-coordinates.
        height (int): Height of the map (default=80).
        width (int): Width of the map (default=80).
        radius (int): Radius within which the value decreases (default=25).
        max_value (float): Maximum value at the center (default=0.0025).
        min_value (float): Minimum value at the edge of the radius (default=2.5e-6).

    Returns:
        torch.Tensor: A map of shape (B, 1, height, width).
    """
    # Initialize the map with zeros
    map_tensor = torch.zeros((B, 1, height, width))+2.5e-6

    # Create a grid for distance calculation
    y_grid = torch.arange(height).view(1, height, 1).repeat(1, 1, width)
    x_grid = torch.arange(width).view(1, 1, width).repeat(1, height, 1)

    # Iterate over each batch
    for b in range(B):
        # Extract coordinates for the current batch
        h, w = hidx_B[b].item(), widx_B[b].item()

        # Compute the distance from the (h, w) point
        distance = torch.sqrt((y_grid - h)**2 + (x_grid - w)**2)

        # Cosine decay formula
        decay = 0.5 * (1 + torch.cos(torch.clamp(distance / radius * torch.pi, max=torch.pi)))
        #decay = torch.exp(-distance / radius)

        # Scale decay to the specified range [min_value, max_value]
        scaled_decay = min_value + (max_value - min_value) * decay
        map_tensor[b, 0] = torch.where(distance <= radius, scaled_decay, min_value)

    return map_tensor

def smooth_map_with_gaussian(map_tensor, sigma=3):
    """
    Smooths the input map using a Gaussian filter.

    Parameters:
        map_tensor (torch.Tensor): The map tensor of shape (B, 1, H, W).
        sigma (int): Standard deviation for the Gaussian filter.

    Returns:
        torch.Tensor: Smoothed map.
    """
    # Define the size of the Gaussian kernel based on sigma
    kernel_size = 2 * int(3 * sigma) + 1  # Covers ±3 standard deviations
    # Create 1D Gaussian kernel
    x = torch.arange(kernel_size) - kernel_size // 2
    gauss_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    gauss_1d = gauss_1d / gauss_1d.sum()  # Normalize

    # Create 2D Gaussian kernel
    gauss_2d = torch.outer(gauss_1d, gauss_1d).unsqueeze(0).unsqueeze(0)

    B = map_tensor.shape[0]
    gauss_2d = gauss_2d.expand(1, 1, kernel_size, kernel_size)
    gauss_2d = gauss_2d.to(map_tensor.device)

    # Apply Gaussian filter to each map
    smoothed_map = F.conv2d(map_tensor, gauss_2d, padding=kernel_size // 2, groups=1)
    return smoothed_map

class CompressNet(nn.Module):
    def __init__(self, cfg):
        super(CompressNet, self).__init__()
        if cfg.MODEL.saliency_net == 'fovsimple':
            self.conv_last = nn.Conv2d(24,1,kernel_size=1,padding=0,stride=1)
        else:
            self.conv_last = nn.Conv2d(256,1,kernel_size=1,padding=0,stride=1)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.act(x)
        out = self.conv_last(x)
        return out

class SegmentationModuleBase(nn.Module):
    def __init__(self):
        super(SegmentationModuleBase, self).__init__()

    def pixel_acc(self, pred_all, label_all):
        # accuracy with forground class
        torch.set_printoptions(threshold=10000)
        bs = pred_all.shape[0]
        acc_accu = 0.
        for i in range(bs):
            pred, label= pred_all[i:i+1, :,:], label_all[i:i+1, :, :] #pred: BxCxHxW  label: BxHxW
            _, preds = torch.max(pred, dim=1) #BxHxW
            valid = (label < 50).long()
            valid1 = (preds < 50).long()
            acc_sum = torch.sum(valid * (preds == label).long())   # this is the intersectory pixels based on class
            pixel_sum = torch.sum(valid)   # this is the summation of number of ground truth pixel
            pixel_sum1 = torch.sum(valid1)  # this is the summation of number of predicted true pixel
            pixel_sum_final = ((valid + valid1) > 0).sum().long()
            acc = acc_sum.float() / (pixel_sum_final.float() + 1e-10)
            acc_accu += acc
            
        return acc_accu/bs
    
    def fg_bin_pixel_acc(self, pred_all, label_all):
        # accuracy with foreground binary
        torch.set_printoptions(threshold=10000)
        bs = pred_all.shape[0]
        acc_accu = 0.
        for i in range(bs):
            pred, label= pred_all[i:i+1, :,:], label_all[i:i+1, :, :] #pred: BxCxHxW  label: BxHxW

            _, preds = torch.max(pred, dim=1) #BxHxW

            valid = (label < 50).long()
            valid1 = (preds < 50).long()
            acc_sum = torch.sum(valid * (valid == valid1).long())   # this is the intersectory pixels based on binary

            pixel_sum = torch.sum(valid)   # this is the summation of number of ground truth pixel
            pixel_sum1 = torch.sum(valid1)  # this is the summation of number of predicted true pixel
            pixel_sum_final = ((valid + valid1)> 0).sum().long()
            acc = acc_sum.float() / (pixel_sum_final.float() + 1e-10)
            acc_accu += acc
            
        return acc_accu/bs
    
    def fbg_cls_pixel_acc(self, pred_all, label_all):
        # accuracy with foreground binary
        torch.set_printoptions(threshold=10000)
        bs = pred_all.shape[0]
        acc_accu = 0.
        for i in range(bs):
            pred, label= pred_all[i:i+1, :,:], label_all[i:i+1, :, :] #pred: BxCxHxW  label: BxHxW
            _, preds = torch.max(pred, dim=1) #BxHxW

            pred_unique = torch.unique(preds)

            label_unique = torch.unique(label)

            valid_fg = (label < 50).long()
            valid1_fg = (preds < 50).long()
            acc_sum_fg = torch.sum(valid_fg * (label == preds).long())   # this is the intersectory pixels based on binary

            pixel_sum_final_fg = ((valid_fg + valid1_fg)> 0).sum().long()
            acc_fg = acc_sum_fg.float() / (pixel_sum_final_fg.float() + 1e-10)

            valid_bg = (label == 50).long()
            valid1_bg = (preds== 50).long()
            acc_sum_bg = torch.sum(valid_bg * (label == preds).long())   # this is the intersectory pixels based on binary

            pixel_sum_final_bg = ((valid_bg + valid1_bg)> 0).sum().long()
            acc_bg = acc_sum_bg.float() / (pixel_sum_final_bg.float() + 1e-10)

            acc_accu += acc_fg*0.5+acc_bg*0.5
            
        return acc_accu/bs
    
    def fbg_bin_pixel_acc(self, pred_all, label_all):
        # accuracy with foreground binary
        torch.set_printoptions(threshold=10000)
        bs = pred_all.shape[0]
        acc_accu = 0.
        for i in range(bs):
            pred, label= pred_all[i:i+1, :,:], label_all[i:i+1, :, :] #pred: BxCxHxW  label: BxHxW
            
            _, preds = torch.max(pred, dim=1) #BxHxW
            valid_fg = (label < 50).long()
            valid1_fg = (preds < 50).long()
            acc_sum_fg = torch.sum(valid_fg * (valid_fg == valid1_fg).long())   # this is the intersectory pixels based on binary
            
            pixel_sum_final_fg = ((valid_fg + valid1_fg)> 0).sum().long()
            acc_fg = acc_sum_fg.float() / (pixel_sum_final_fg.float() + 1e-10)
            
            valid_bg = (label == 50).long()
            valid1_bg = (preds == 50).long()
            acc_sum_bg = torch.sum(valid_bg * (valid_bg == valid1_bg).long())   # this is the intersectory pixels based on binary
            
            pixel_sum_final_bg = ((valid_bg + valid1_bg)> 0).sum().long()
            acc_bg = acc_sum_bg.float() / (pixel_sum_final_bg.float() + 1e-10)
            acc_accu += acc_fg*0.5+acc_bg*0.5
            
        return acc_accu/bs
    
class DeformSegmentationModule(SegmentationModuleBase):
    def __init__(self, net_encoder, net_decoder, net_saliency, net_compress, crit, cfg, deep_sup_scale=None):
        super(DeformSegmentationModule, self).__init__()
        self.encoder = net_encoder
        self.decoder = net_decoder
        self.localization = net_saliency
        self.crit = DiceLoss('multiclass')

        if cfg.TRAIN.opt_deform_LabelEdge or cfg.TRAIN.deform_joint_loss:
            self.crit_mse = nn.MSELoss()
        self.cfg = cfg
        self.deep_sup_scale = deep_sup_scale
        self.print_original_y = True
        self.net_compress = net_compress
        if self.cfg.MODEL.saliency_output_size_short == 0:
            self.grid_size_x = cfg.TRAIN.saliency_input_size[0]
        else:
            self.grid_size_x = self.cfg.MODEL.saliency_output_size_short
        self.grid_size_y = cfg.TRAIN.saliency_input_size[1] // (cfg.TRAIN.saliency_input_size[0]//self.grid_size_x)
        self.padding_size_x = self.cfg.MODEL.gaussian_radius
        if self.cfg.MODEL.gaussian_ap == 0.0:
            gaussian_ap = cfg.TRAIN.saliency_input_size[1] // cfg.TRAIN.saliency_input_size[0]
        else:
            gaussian_ap = self.cfg.MODEL.gaussian_ap
        self.padding_size_y = int(gaussian_ap * self.padding_size_x)
        self.global_size_x = self.grid_size_x+2*self.padding_size_x
        self.global_size_y = self.grid_size_y+2*self.padding_size_y
        self.input_size = cfg.TRAIN.saliency_input_size
        self.input_size_net = cfg.TRAIN.task_input_size
        self.input_size_net_eval = cfg.TRAIN.task_input_size_eval
        if len(self.input_size_net_eval) == 0:
            self.input_size_net_infer = self.input_size_net
        else:
            self.input_size_net_infer = self.input_size_net_eval
        gaussian_weights = torch.FloatTensor(makeGaussian(2*self.padding_size_x+1, fwhm = self.cfg.MODEL.gaussian_radius)) # TODO: redo seneitivity experiments on gaussian radius as corrected fwhm (effective gaussian radius)
        gaussian_weights = b_imresize(gaussian_weights.unsqueeze(0).unsqueeze(0), (2*self.padding_size_x+1,2*self.padding_size_y+1), interp='bilinear')
        gaussian_weights = gaussian_weights.squeeze(0).squeeze(0)

        self.filter = nn.Conv2d(1, 1, kernel_size=(2*self.padding_size_x+1,2*self.padding_size_y+1),bias=False)
        self.filter.weight[0].data[:,:,:] = gaussian_weights

        self.P_basis = torch.zeros(2,self.grid_size_x+2*self.padding_size_x, self.grid_size_y+2*self.padding_size_y).cuda()
        # initialization of u(x,y),v(x,y) range from 0 to 1
        for k in range(2):
            for i in range(self.global_size_x):
                for j in range(self.global_size_y):
                    self.P_basis[k,i,j] = k*(i-self.padding_size_x)/(self.grid_size_x-1.0)+(1.0-k)*(j-self.padding_size_y)/(self.grid_size_y-1.0)

        self.save_print_grad = [{'saliency_grad': 0.0, 'check1_grad': 0.0, 'check2_grad': 0.0} for _ in range(cfg.TRAIN.num_gpus)]
    
    def unorm(self, img):
        if 'GLEASON' in self.cfg.DATASET.list_train:
            mean=[0.748, 0.611, 0.823]
            std=[0.146, 0.245, 0.119]
        elif 'Digest' in self.cfg.DATASET.list_train:
            mean=[0.816, 0.697, 0.792]
            std=[0.160, 0.277, 0.198]
        elif 'ADE' in self.cfg.DATASET.list_train:
            mean=[0.485, 0.456, 0.406]
            std=[0.229, 0.224, 0.225]
        elif 'CITYSCAPE' in self.cfg.DATASET.list_train or 'Cityscape' in self.cfg.DATASET.list_train:
            mean=[0.485, 0.456, 0.406]
            std=[0.229, 0.224, 0.225]
        elif 'histo' in self.cfg.DATASET.list_train:
            mean=[0.8223, 0.7783, 0.7847]
            std=[0.210, 0.216, 0.241]
        elif 'DeepGlob' in self.cfg.DATASET.list_train or 'deepglob' in self.cfg.DATASET.root_dataset:
            mean=[0.282, 0.379, 0.408]
            std=[0.089, 0.101, 0.127]
        elif 'Face_single_example' in self.cfg.DATASET.root_dataset or 'Face_single_example' in self.cfg.DATASET.list_train:
            mean=[0.282, 0.379, 0.408]
            std=[0.089, 0.101, 0.127]
        elif 'Histo' in self.cfg.DATASET.root_dataset or 'histomri' in self.cfg.DATASET.list_train or 'histomri' in self.cfg.DATASET.root_dataset:
            mean=[0.8223, 0.7783, 0.7847]
            std=[0.210, 0.216, 0.241]
        else:
            raise Exception('Unknown root for normalisation!')
        for t, m, s in zip(img, mean, std):
            t.mul_(s).add_(m)
        return img
    
    def re_initialise(self, cfg, this_size):# dealing with varying input image size such as pcahisto dataset
        this_size_short = min(this_size)
        this_size_long = max(this_size)
        scale_task_rate_1 = this_size_short // min(cfg.TRAIN.dynamic_task_input)
        scale_task_rate_2 = this_size_long // max(cfg.TRAIN.dynamic_task_input)
        scale_task_size_1 = tuple([int(x//scale_task_rate_1) for x in this_size])
        scale_task_size_2 = tuple([int(x//scale_task_rate_2) for x in this_size])
        if scale_task_size_1[0]*scale_task_size_1[1] < scale_task_size_2[0]*scale_task_size_2[1]:
            scale_task_size = scale_task_size_1
        else:
            scale_task_size = scale_task_size_2

        cfg.TRAIN.task_input_size = scale_task_size
        cfg.TRAIN.saliency_input_size = tuple([int(x*cfg.TRAIN.dynamic_saliency_relative_size) for x in scale_task_size])

        if self.cfg.MODEL.saliency_output_size_short == 0:
            self.grid_size_x = cfg.TRAIN.saliency_input_size[0]
        else:
            self.grid_size_x = self.cfg.MODEL.saliency_output_size_short
        self.grid_size_y = cfg.TRAIN.saliency_input_size[1] // (cfg.TRAIN.saliency_input_size[0]//self.grid_size_x)
        self.global_size_x = self.grid_size_x+2*self.padding_size_x
        self.global_size_y = self.grid_size_y+2*self.padding_size_y

        self.input_size = cfg.TRAIN.saliency_input_size
        self.input_size_net = cfg.TRAIN.task_input_size

        if len(self.input_size_net_eval) == 0:
            self.input_size_net_infer = self.input_size_net
        else:
            self.input_size_net_infer = self.input_size_net_eval
        self.P_basis = torch.zeros(2,self.grid_size_x+2*self.padding_size_x, self.grid_size_y+2*self.padding_size_y).cuda()
        # initialization of u(x,y),v(x,y) range from 0 to 1
        for k in range(2):
            for i in range(self.global_size_x):
                for j in range(self.global_size_y):
                    self.P_basis[k,i,j] = k*(i-self.padding_size_x)/(self.grid_size_x-1.0)+(1.0-k)*(j-self.padding_size_y)/(self.grid_size_y-1.0)

    def create_grid(self, x, segSize=None, x_inv=None):
        P = torch.autograd.Variable(torch.zeros(1,2,self.grid_size_x+2*self.padding_size_x, self.grid_size_y+2*self.padding_size_y, device=x.device),requires_grad=False)

        P[0,:,:,:] = self.P_basis.to(x.device) # [1,2,w,h], 2 corresponds to u(x,y) and v(x,y)
        P = P.expand(x.size(0),2,self.grid_size_x+2*self.padding_size_x, self.grid_size_y+2*self.padding_size_y)
        # input x is saliency map xs
        x_cat = torch.cat((x,x),1)
        # EXPLAIN: denominator of learn to downsample Eq. (3)
        p_filter = self.filter(x)
        x_mul = torch.mul(P,x_cat).view(-1,1,self.global_size_x,self.global_size_y)
        all_filter = self.filter(x_mul).view(-1,2,self.grid_size_x,self.grid_size_y)
        # EXPLAIN: numerator of learn to downsample Eq. (3)
        x_filter = all_filter[:,0,:,:].contiguous().view(-1,1,self.grid_size_x,self.grid_size_y)
        y_filter = all_filter[:,1,:,:].contiguous().view(-1,1,self.grid_size_x,self.grid_size_y)
        # EXPLAIN: learn to downsample Eq. (3)
        x_filter = x_filter/p_filter
        y_filter = y_filter/p_filter
        # EXPLAIN: fit F.grid_sample format (coordibates in the range [-1,1])
        xgrids = x_filter*2-1
        ygrids = y_filter*2-1
        xgrids = torch.clamp(xgrids,min=-1,max=1)
        ygrids = torch.clamp(ygrids,min=-1,max=1)
        # EXPLAIN: reshape
        xgrids = xgrids.view(-1,1,self.grid_size_x,self.grid_size_y)
        ygrids = ygrids.view(-1,1,self.grid_size_x,self.grid_size_y)
        grid = torch.cat((xgrids,ygrids),1)

        if len(self.input_size_net_eval) != 0 and segSize is not None:# inference
            
            grid = nn.Upsample(size=self.input_size_net_infer, mode='bilinear')(grid)
        else:
            grid = nn.Upsample(size=self.input_size_net, mode='bilinear')(grid)
        # EXPLAIN: grid_y for downsampling label y, to handle segmentation architectures whose prediction are not same size with input x
        if segSize is None:# training
            # enter here
            grid_y = nn.Upsample(size=tuple(np.array(self.input_size_net)//self.cfg.DATASET.segm_downsampling_rate), mode='bilinear')(grid)
        else:# inference
            grid_y = nn.Upsample(size=tuple(np.array(self.input_size_net_infer)), mode='bilinear')(grid)

        grid = torch.transpose(grid,1,2)
        grid = torch.transpose(grid,2,3)

        grid_y = torch.transpose(grid_y,1,2)
        grid_y = torch.transpose(grid_y,2,3)

        #### inverse deformation
        if segSize is not None and x_inv is not None:
            grid_reorder = grid.permute(3,0,1,2)
            grid_inv = torch.autograd.Variable(torch.zeros((2,grid_reorder.shape[1],segSize[0],segSize[1]), device=grid_reorder.device))
            grid_inv[:] = float('nan')
            u_cor = (((grid_reorder[0,:,:,:]+1)/2)*(segSize[1]-1)).int().long().view(grid_reorder.shape[1],-1)
            v_cor = (((grid_reorder[1,:,:,:]+1)/2)*(segSize[0]-1)).int().long().view(grid_reorder.shape[1],-1)
            x_cor = torch.arange(0,grid_reorder.shape[3], device=grid_reorder.device).unsqueeze(0).expand((grid_reorder.shape[2],grid_reorder.shape[3])).reshape(-1)
            x_cor = x_cor.unsqueeze(0).expand(u_cor.shape[0],-1).float()
            y_cor = torch.arange(0,grid_reorder.shape[2], device=grid_reorder.device).unsqueeze(-1).expand((grid_reorder.shape[2],grid_reorder.shape[3])).reshape(-1)
            y_cor = y_cor.unsqueeze(0).expand(u_cor.shape[0],-1).float()
            grid_inv[0][torch.arange(grid_reorder.shape[1]).unsqueeze(-1),v_cor,u_cor] = torch.autograd.Variable(x_cor)
            grid_inv[1][torch.arange(grid_reorder.shape[1]).unsqueeze(-1),v_cor,u_cor] = torch.autograd.Variable(y_cor)
            grid_inv[0] = grid_inv[0]/grid_reorder.shape[3]*2-1
            grid_inv[1] = grid_inv[1]/grid_reorder.shape[2]*2-1
            grid_inv = grid_inv.permute(1,2,3,0)
            return grid, grid_inv
        else:
            return grid, grid_y

    def ignore_label(self, label, ignore_indexs):
        label = np.array(label)
        temp = label.copy()
        for k in ignore_indexs:
            label[temp == k] = 0
        return label

    def forward(self, feed_dict, *, writer=None, segSize=None, F_Xlr_acc_map=False, count=None, epoch=None, feed_dict_info=None, feed_batch_count=None, cur_iter=None, is_inference=False, rank = None):
        #pdb.set_trace()
        upsample = self.cfg.MODEL.upsample
        # EXPLAIN: re initialise apply only when input image has varying size, e.g. pcahisto dataset
        if self.cfg.TRAIN.dynamic_task_input[0] != 1:
            print('asdfasdfasdf\n\n\n\n')
            this_size = tuple(feed_dict['img_data'].shape[-2:])
            print('this_size: {}'.format(this_size))
            self.re_initialise(self.cfg, this_size)
            print('task_input_size after re_initialise: {}'.format(self.input_size_net))
            print('saliency_input_size after re_initialise: {}'.format(self.input_size))

        # EXPLAIN: for each high-resolution image X
        x = feed_dict['img_data']
        _, _, H_HS, W_HS = x.shape
        t = time.time()
        ori_size = (x.shape[-2],x.shape[-1])
        ######################## enter here!!!!!!!!!!!!!!!!!!##########
        x_Bx2 = feed_dict['focus_point']
        B, _ = x_Bx2.shape
        HS, WS = self.input_size
        max_dist = np.sqrt(HS ** 2 + WS ** 2)

        hidx_B = (x_Bx2[:, 0] * (HS - 1))
        widx_B = (x_Bx2[:, 1] * (WS - 1))

        grid_mtx_Bx2xHxW = gen_grid_mtx_2xHxW(HS, WS, device='cuda').unsqueeze(0).repeat(B, 1, 1, 1)
        dist_BxHxW = torch.sqrt((grid_mtx_Bx2xHxW[:, 0, :, :] - hidx_B[:, None, None]) ** 2 + (grid_mtx_Bx2xHxW[:, 1, :, :] - widx_B[:, None, None]) ** 2)
        focusmap_Bx1xHxW = (dist_BxHxW / max_dist).unsqueeze(1)**2

        fp_tensor = torch.zeros((B, 1, HS, WS)).to(x)
        for b in range(B):
            fp_tensor[b, 0, hidx_B.round().int(), widx_B.round().int()] = 1
        ###############################################################
        # EXPLAIN: compute its lower resolution version Xlr
        x_low = b_imresize(x, self.input_size, interp='bilinear')

        #############enter here!!!!##################################################
        x_low = torch.cat((x_low, focusmap_Bx1xHxW), dim=1)
        x_low = torch.cat((x_low, focusmap_Bx1xHxW), dim=1)
        #x_low = torch.cat((x_low, fp_tensor), dim=1)
        ###############################################################

        epoch = self.cfg.TRAIN.global_epoch

        xs = self.localization(x_low)

        xs = self.net_compress(xs)

        xs = nn.Upsample(size=(self.grid_size_x,self.grid_size_y), mode='bilinear')(xs)
        xs = xs.view(-1,self.grid_size_x*self.grid_size_y) # N,1,W,H



        xs = nn.Softmax()(xs) # N,W*H
        assert not torch.isnan(xs).any(), "xs contains NaN values!"

        xs = xs.view(-1,1,self.grid_size_x,self.grid_size_y) #saliency map

        y = feed_dict['seg_label'].clone() 
        xs_our = xs.clone()
        y_temp = y.clone()
        if len(y_temp.shape) == 3:
            y_temp = y_temp.unsqueeze(0)
        xs_target_our = F.interpolate(y_temp, size=(self.grid_size_x,self.grid_size_y), mode='area') 

        # EXPLAIN: calculate the target deformation map dt = fedge(fgaus(Ylr)) from the uniformly downsampled segmentation labelYlr
        if self.cfg.MODEL.gt_gradient or (self.cfg.MODEL.uniform_sample == 'BI' and self.cfg.DATASET.num_class == 2):
        # EXPLAIN: if motivational study (gt_gradient) or uniform downsample (uniform_sample == 'BI')
            #print('aaaaaaaaaaaaaaaaaaaaaaa\n\n\n\n\n\n')
            xsc = xs.clone().detach()
            for j in range(y.shape[0]):
                if segSize is not None:
                    # i.e. if training
                    (y_j_dist, _) = np.histogram(y[j].cpu(), bins=2, range=(0, 1))
                if self.cfg.MODEL.fix_gt_gradient and not (self.cfg.MODEL.uniform_sample == 'BI' and self.cfg.DATASET.num_class == 2):
                    # EXPLAIN: for motivational study: simulating a set "edge-based" samplers each at different sampling density around edge
                    y_clone = y.clone().cpu()
                    if self.cfg.MODEL.ignore_gt_labels != []:
                        y_clone = torch.tensor(self.ignore_label(y_clone, self.cfg.MODEL.ignore_gt_labels))
                    y_norm = (y_clone[j] - y_clone[j].min()).float()/(y_clone[j].max() - y_clone[j].min()).float()
                    y_low = b_imresize(y_norm.unsqueeze(0).unsqueeze(0).float(), self.input_size, interp='bilinear') # [N,1,W,H]
                    y_gradient = y_low.clone() # [N,C,W,H]
                    y_low_cpu = y_low.cpu()
                    y_low_img_ay = np.array((y_low_cpu[0][0]*255)).astype(np.uint8)
                    y_low_img = Image.fromarray(y_low_img_ay, 'L')
                    # apply gaussian blur to avoid not having enough saliency for sampling
                    y_low_img = y_low_img.filter(ImageFilter.GaussianBlur(radius=self.cfg.MODEL.gt_grad_gaussian_blur_r )) # default radius=2
                    y_low_Edges = y_low_img.filter(ImageFilter.FIND_EDGES)
                    y_gradient[0][0] = torch.tensor(np.array(y_low_Edges.convert('L'))/255.).to(y_low.device)
                    xs_j = nn.Upsample(size=(self.grid_size_x,self.grid_size_y), mode='bilinear')(y_gradient)
                    xsc_j = xs_j[0] # 1,W,H
                    xsc[j] = xsc_j
                if segSize is not None and y_j_dist[1]/y_j_dist.sum() <= 0.001 and self.cfg.DATASET.binary_class != -1:
                    # exclude corner case
                    print('y_{} do not have enough forground class, skip this sample\n'.format(j))
                    if self.cfg.VAL.y_sampled_reverse:
                        return None, None, None, None
                    else:
                        return None, None, None

            if self.cfg.TRAIN.deform_zero_bound:
                xsc_mask = xsc.clone()*0.0
                bound = self.cfg.TRAIN.deform_zero_bound_factor
                xsc_mask[:,:,1*bound:-1*bound,1*bound:-1*bound] += 1.0
                xsc *= xsc_mask
            xs.data = xsc.data.to(xs.device)
        elif self.cfg.TRAIN.opt_deform_LabelEdge or self.cfg.TRAIN.deform_joint_loss:
        # EXPLAIN: if ours - calculate the target deformation map xs_target for edge_loss
            # enter here!!!!!
            xs_target = xs.clone().detach()
            for j in range(y.shape[0]):
                (y_j_dist, _) = np.histogram(y[j].cpu(), bins=2, range=(0, 1))
                if not (self.cfg.MODEL.uniform_sample == 'BI' and self.cfg.DATASET.num_class == 2):
                    # enter here!!!!!!!!!!!!!
                    y_norm = (y[j] - y[j].min()).float()/(y[j].max() - y[j].min()).float()

                    y_low = b_imresize(y_norm.unsqueeze(0).float(), self.input_size, interp='bilinear') # [N,1,W,H]

                    y_gradient = y_low.clone() # [N,C,W,H]
                    y_low_cpu = y_low.cpu()


                    y_low_img_ay = np.array((y_low_cpu[0][0]*255)).astype(np.uint8)
                    y_low_img = Image.fromarray(y_low_img_ay, 'L')
                    y_low_img = y_low_img.filter(ImageFilter.GaussianBlur(radius=self.cfg.MODEL.gt_grad_gaussian_blur_r )) # default radius=2
                    y_low_Edges = y_low_img.filter(ImageFilter.FIND_EDGES)
                    y_gradient[0][0] = torch.tensor(np.array(y_low_Edges.convert('L'))/255.).to(y_low.device)
                    xs_j = nn.Upsample(size=(xs_target.shape[-2],xs_target.shape[-1]), mode='bilinear')(y_gradient)
                    (xs_j_dist, _) = np.histogram(xs_j.cpu(), bins=10, range=(0, 1))
                    xsc_j = xs_j[0] # 1,W,H
                    if self.cfg.TRAIN.opt_deform_LabelEdge_softmax:
                        xsc_j = xsc_j.view(1,xs_target.shape[-2]*xs_target.shape[-1])
                        xsc_j = nn.Softmax()(xsc_j)
                    xs_target[j] = xsc_j.view(1,xs_target.shape[-2],xs_target.shape[-1])
                    
                elif y_j_dist[1]/y_j_dist.sum() <= 0.001 and self.cfg.DATASET.binary_class != -1:
                    print('y_{} do not have enough forground class, skip this sample\n'.format(j))
                    if segSize is not None:
                        if self.cfg.VAL.y_sampled_reverse:
                            return None, None, None, None
                        else:
                            return None, None, None
            if self.cfg.TRAIN.deform_zero_bound:
                xs_target_mask = xs_target.clone()*0.0
                bound = self.cfg.TRAIN.deform_zero_bound_factor
                xs_target_mask[:,:,1*bound:-1*bound,1*bound:-1*bound] += 1.0
                xs_target *= xs_target_mask
        #pdb.set_trace()
        # EXPLAIN: pad to avoid boundary artifact following A. Recasens et,al. (2018)
        if self.cfg.MODEL.uniform_sample != '':
            # not used
            xs = xs*0 + 1.0/(self.grid_size_x*self.grid_size_y)
        if self.cfg.TRAIN.def_saliency_pad_mode == 'replication':
            # enter here!!!!!!!!!!!!!!!!!
            xs_hm = nn.ReplicationPad2d((self.padding_size_y, self.padding_size_y, self.padding_size_x, self.padding_size_x))(xs) # padding by replicate the edges in the map
        elif self.cfg.TRAIN.def_saliency_pad_mode == 'reflect':
            xs_hm = F.pad(xs, (self.padding_size_y, self.padding_size_y, self.padding_size_x, self.padding_size_x), mode='reflect')
        elif self.cfg.TRAIN.def_saliency_pad_mode == 'zero':
            xs_hm = F.pad(xs, (self.padding_size_y, self.padding_size_y, self.padding_size_x, self.padding_size_x), mode='constant')
        # EXPLAIN: if training

        if segSize is None:
            # enter here!!!

            # EXPLAIN: pretraining trick following A. Recasens et,al. (2018)
            N_pretraining = self.cfg.TRAIN.deform_pretrain
            epoch = self.cfg.TRAIN.global_epoch
            #print(N_pretraining, epoch)   # 100 1
            #exit()
            if self.cfg.TRAIN.deform_pretrain_bol or (epoch>=N_pretraining and (epoch<self.cfg.TRAIN.smooth_deform_2nd_start or epoch>self.cfg.TRAIN.smooth_deform_2nd_end)):
                # enter here!!!!!
                p=1 # non-pretain stage: no random size pooling to x_sampled
            else:
                p=0 # pretrain stage: random size pooling to x_sampled
            

            # EXPLAIN: construct the deformed sampler Gd (Eq. 3)

            grid, grid_y = self.create_grid(xs_hm)            

            ###################################
            # plotting
            segSize = (feed_dict['seg_label'].shape[-2], feed_dict['seg_label'].shape[-1])
            xs_inv = 1-xs_hm
            _, grid_inv = self.create_grid(xs_hm, segSize=segSize, x_inv=xs_inv)
            pred_sampled_unfilled_mask_2d = torch.isnan(grid_inv[:,:,:,0])
            if self.cfg.DATASET.grid_path != '':
                grid_img = Image.open(self.cfg.DATASET.grid_path).convert('RGB')
                grid_resized = grid_img.resize(segSize, Image.BILINEAR)
                del grid_img

                grid_resized = np.float32(np.array(grid_resized)) / 255.
                grid_resized = grid_resized.transpose((2, 0, 1))
                grid_resized = torch.from_numpy(grid_resized.copy())

                grid_resized = torch.unsqueeze(grid_resized, 0).expand(grid.shape[0],grid_resized.shape[-3],grid_resized.shape[-2],grid_resized.shape[-1])
                grid_resized = grid_resized.to(feed_dict['seg_label'].device)

                grid_output = F.grid_sample(grid_resized, grid).detach()
                del grid_resized
            ###################################    

            if self.cfg.MODEL.loss_at_high_res or (upsample):
                #use this to calculate inv_grid for upsample
                # noe there here
                xs_inv = 1-xs_hm
                _, grid_inv_train = self.create_grid(xs_hm, segSize=tuple(np.array(ori_size)//self.cfg.DATASET.segm_downsampling_rate), x_inv=xs_inv)
            # EXPLAIN: during training the labelY is downsampled with the same deformed sampler to get Yˆ = Gd(Y,d) (i.e. y_low)
            if self.cfg.MODEL.uniform_sample == 'BI':
                #print('aaaaaaaaaaaaaaaaaaaaaa\n\n\n')
                y_sampled = nn.Upsample(size=tuple(np.array(self.input_size_net)//self.cfg.DATASET.segm_downsampling_rate), mode='bilinear')(y.float().unsqueeze(1)).long().squeeze(1)
            else:
                # enter here!!!!!!!!!!!!, sample y based on grid_y
                y_sampled = F.grid_sample(y.float(), grid_y).squeeze(1)

            # EXPLAIN: calculate the mse loss of target object
            if self.cfg.TRAIN.opt_deform_LabelEdge or self.cfg.TRAIN.deform_joint_loss:
                # enter here!!!!!!!!!!!!!!
                assert (xs.shape == xs_target.shape), "xs shape ({}) not equvelent to xs_target shape ({})\n".format(xs.shape, xs_target.shape)
                if self.cfg.TRAIN.opt_deform_LabelEdge_norm:
                    # enter here!!!!                    
                    # mse loss
                    xs_our_norm = ((xs_our - xs_our.min()) / (xs_our.max() - xs_our.min()))
                    xs_target_our_norm = ((xs_target_our - xs_target_our.min()) / (xs_target_our.max() - xs_target_our.min()))
                    edge_loss = 0.05*self.crit_mse(xs_our_norm, xs_target_our_norm)
                    assert not torch.isnan(edge_loss).any(), "edge_loss contains NaN values!"
                    edge_acc = self.pixel_acc(xs_our_norm.long(), xs_target_our_norm.long())

                else:
                    edge_loss = self.crit_mse(xs, xs_target)
                    edge_acc = self.pixel_acc(xs.long(), xs_target.long())
                edge_loss *= self.cfg.TRAIN.edge_loss_scale
                # EXPLAIN: for staged training when traing with only the edge loss, not used
                if self.cfg.TRAIN.opt_deform_LabelEdge and epoch >= self.cfg.TRAIN.fix_seg_start_epoch and epoch <= self.cfg.TRAIN.fix_seg_end_epoch:
                    return edge_loss, edge_acc, edge_loss

            # EXPLAIN: computes the downsampled image X^ =Gd(X,d)
            if self.cfg.MODEL.uniform_sample == 'BI':
                # not used
                x_sampled = nn.Upsample(size=self.input_size_net, mode='bilinear')(x)
            else:
                # enter here!!!!!!!!!!!!!!
                x_sampled = F.grid_sample(x, grid)
                deform_image = x_sampled.clone()
            # EXPLAIN: pretraining trick following A. Recasens et,al. (2018)
            if random.random()>p:
                # not used
                min_saliency_len = min(self.input_size)
                s = random.randint(min_saliency_len//3, min_saliency_len)
                x_sampled = nn.AdaptiveAvgPool2d((s,s))(x_sampled)
                x_sampled = nn.Upsample(size=self.input_size_net,mode='bilinear')(x_sampled)
        
            # EXPLAIN: The downsampled image X^ is then fed into the segmentation network to
            # estimate the corresponding segmentation probabilities Pˆ =Sϕ(Xˆ)
            if self.deep_sup_scale is not None: # use deep supervision technique
                # not used
                (pred, pred_deepsup) = self.decoder(self.encoder(x_sampled, return_feature_maps=True))
            else:
                #enter here!!!!
                pred = self.decoder(self.encoder(x_sampled, return_feature_maps=True))
                assert not torch.isnan(pred).any(), "pred contains NaN values!"
            torch.cuda.reset_max_memory_allocated(0)
            # EXPLAIN: ablation, if calculate loss at high resolution, inverse upsample the prediction to high-res space
            if self.cfg.MODEL.loss_at_high_res and self.cfg.MODEL.uniform_sample == 'BI':
                # not used
                pred_sampled_train = nn.Upsample(size=ori_size, mode='bilinear')(pred)
            elif self.cfg.MODEL.loss_at_high_res or (upsample):
                # if upsample
                unfilled_mask_2d = torch.isnan(grid_inv_train[:,:,:,0])
                grid_inv_train[torch.isnan(grid_inv_train)] = 0
                pred_sampled_train = F.grid_sample(pred, grid_inv_train.float())
                pred_sampled_train[unfilled_mask_2d.unsqueeze(1).expand(pred_sampled_train.shape)] = float('nan')
                for n in range(pred_sampled_train.shape[0]):
                    pred_sampled_train[n] = fillMissingValues_tensor(pred_sampled_train[n], interp_mode=self.cfg.MODEL.rev_deform_interp)

            if self.deep_sup_scale is not None: # use deep supervision technique
                # not used
                pred, pred_deepsup = pred, pred_deepsup
            if self.cfg.MODEL.loss_at_high_res:
                # not used
                pred,image_output,hm,_,pred_sampled = pred,x_sampled,xs,y_sampled,pred_sampled_train
            else:
                # enter here!!!!!!!!!!!!!!!!!    # change the naming
                if not upsample:
                    pred,image_output,hm,feed_dict['seg_label'], y_hs = pred,x_sampled,xs,y_sampled.long(), feed_dict['seg_label'].squeeze(1)
                else:
                    assert not torch.isnan(pred).any(), "pred contains NaN values!"
                    pred,image_output,hm,feed_dict['seg_label'], y_hs, pred_sampled = pred,x_sampled,xs,y_sampled.long(),feed_dict['seg_label'].squeeze(1),pred_sampled_train
                    assert not torch.isnan(pred).any(), "pred contains NaN values!"

            if self.cfg.MODEL.loss_at_high_res:
                # not used
                del pred_sampled_train
            del y_sampled
            # EXPLAIN: end of training, calculate loss and return
            if self.cfg.MODEL.loss_at_high_res:
                # not used
                pred_sampled[torch.isnan(pred_sampled)] = 0 # assign residual missing with 0 probability
                loss = self.crit(pred_sampled, feed_dict['seg_label'])
            else:
                label1 = feed_dict['seg_label']  # torch.Size([20, 64, 128])  0,1
                ground_truth = label1 * feed_dict['cls_label'][:, :, None].repeat(1, HS, WS).cuda() + (1-label1)*50

                B, _, _, _ = pred.shape
                gt_hs = y_hs * feed_dict['cls_label'][:, :, None].repeat(1, H_HS, W_HS).cuda() + (1-y_hs)*50

                if not is_inference and epoch%10 == 1 and rank == 0:
                    save_dir = os.path.join(self.cfg.DIR, f"train_visual_epoch{epoch}")
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                elif is_inference and rank == 0:
                    save_dir = os.path.join(self.cfg.DIR, f"valid_visual_epoch{epoch}")
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                hm = hm.data
                hm_max, _ = hm.view(hm.shape[0],-1).max(dim=1)
                hm_shape = hm.shape
                hm = (hm.view(hm.shape[0],-1)/hm_max.view(hm.shape[0],-1)).view(hm_shape)
                
                if not upsample and rank == 0:
                    #print(cur_iter)
                    if cur_iter >= 0 and cur_iter <= 100 and (epoch%10 == 1 or is_inference):
                        for i in range(int(B/2)):
                            pred_idx = torch.argmax(pred, dim=1, keepdim=True)  # Shape: [10, 1, 64, 128]
                            pred_mask_image = rgb_img(pred_idx[i, 0])

                            pred_image_name = os.path.join(save_dir, f"iter{cur_iter}_batch{i}_pred.png")

                            pred_mask_image.save(pred_image_name)

                            gt_mask_image = rgb_img(ground_truth[i])
                            gt_image_name = os.path.join(save_dir, f"iter{cur_iter}_batch{i}_gt.png")
                            gt_mask_image.save(gt_image_name)

                            gt_mask_image_hr = rgb_img(gt_hs[i])
                            gt_image_name_hr = os.path.join(save_dir, f"iter{cur_iter}_batch{i}_gt_hr.png")
                            gt_mask_image_hr.save(gt_image_name_hr)

                            deformed_grid = vutils.make_grid(grid_output[i].unsqueeze(0), normalize=True, scale_each=True)
                            Image.fromarray(np.array(grid_output[i].permute(1,2,0).cpu()*255.0).astype(np.uint8)).save(os.path.join(save_dir, f"iter{cur_iter}_batch{i}_grid.png"))
                            
                            xhm = vutils.make_grid(hm[i].unsqueeze(0), normalize=True, scale_each=True)
                            Image.fromarray(np.array(hm[i].squeeze(0).squeeze(0).cpu()*255.0).astype(np.uint8)).save(os.path.join(save_dir, f"iter{cur_iter}_batch{i}_sm.png"))

                            red_spot_mask_2d = torch.zeros(pred_sampled_unfilled_mask_2d[i].shape)

                            red_spot_mask_2d[~pred_sampled_unfilled_mask_2d[i]] = 1

                            # Number of `1`s along the y-direction (sum along columns)
                            num_ones_y = red_spot_mask_2d.sum(dim=1)  # Shape: (64,) - Sum along rows

                            # Number of `1`s along the x-direction (sum along rows)
                            num_ones_x = red_spot_mask_2d.sum(dim=0)  # Shape: (64,) - Sum along columns

                            # Find the max and min counts of `1`s across the y-direction
                            max_ones_y = num_ones_y[num_ones_y > 0].max().item()
                            min_ones_y = num_ones_y[num_ones_y > 0].min().item()

                            # Find the max and min counts of `1`s across the x-direction
                            max_ones_x = num_ones_x[num_ones_x > 0].max().item()
                            min_ones_x = num_ones_x[num_ones_x > 0].min().item()

                            # Print the results
                            red_spot_mask_2d = np.array(red_spot_mask_2d)
                            red_spot_mask_2d_dilate = ndimage.binary_dilation(red_spot_mask_2d, iterations=1)
                            img_sampling_masked = feed_dict['img_data'][i]
                            img_sampling_masked = img_sampling_masked.permute(1, 2, 0).cpu().numpy()
                            img_sampling_masked[:,:,0][red_spot_mask_2d_dilate] = 255.0
                            img_sampling_masked[:,:,1][red_spot_mask_2d_dilate] = 0.0
                            img_sampling_masked[:,:,2][red_spot_mask_2d_dilate] = 0.0
                            Image.fromarray(img_sampling_masked.astype(np.uint8)).save(os.path.join(save_dir, f"iter{cur_iter}_batch{i}_reddot.png"))

                elif rank == 0:
                    gt_hs = y_hs * feed_dict['cls_label'][:, :, None].repeat(1, H_HS, W_HS).cuda() + (1-y_hs)*50
                    if cur_iter >= 0 and cur_iter <= 30:
                        for i in range(int(B/2)):
                            pred_idx = torch.argmax(pred_sampled, dim=1, keepdim=True)  # Shape: [10, 1, 64, 128]
                            pred_mask_image = rgb_img(pred_idx[i, 0])
                            #print("rgb image generated")
                            pred_image_name = os.path.join(save_dir, f"iter{cur_iter}_batch{i}_pred.png")
                            pred_mask_image.save(pred_image_name)
                            gt_mask_image = rgb_img(gt_hs[i])
                            gt_image_name = os.path.join(save_dir, f"iter{cur_iter}_batch{i}_gt.png")
                            gt_mask_image.save(gt_image_name)

                assert not torch.isnan(pred).any(), "pred contains NaN values!"
                assert not torch.isnan(ground_truth).any(), "ground_truth contains NaN values!"
                assert not torch.isinf(pred).any(), "pred contains inf values!"
                assert not torch.isinf(ground_truth).any(), "ground_truth contains inf values!"
                focalloss = FocalLoss(gamma=5.0) # TODO was 15
                focal_loss = focalloss(pred, ground_truth)
                dice_loss = self.crit(pred, ground_truth)               
                assert not torch.isnan(focal_loss).any(), "focal_loss contains NaN values!"
                assert not torch.isnan(dice_loss).any(), "dice_loss contains NaN values!"
                loss = dice_loss + focal_loss

            if self.deep_sup_scale is not None:
                loss_deepsup = self.crit(pred_deepsup, feed_dict['seg_label'])
                loss = loss + loss_deepsup * self.deep_sup_scale
            if self.cfg.TRAIN.deform_joint_loss:
                # enter here!!!!
                loss = loss + edge_loss
            if self.cfg.MODEL.loss_at_high_res:
                acc = self.pixel_acc(pred_sampled, feed_dict['seg_label'])
            else:
                ##################################### enter here!!!!############################
                if not (upsample):
                    acc = self.pixel_acc(pred, ground_truth)
                    acc_bin_fg = self.fg_bin_pixel_acc(pred, ground_truth)
                    acc_cls_fbg = self.fbg_cls_pixel_acc(pred, ground_truth)
                    acc_bin_fbg = self.fbg_bin_pixel_acc(pred, ground_truth)
                else:
                    acc = self.pixel_acc(pred_sampled, gt_hs)
                    acc_bin_fg = self.fg_bin_pixel_acc(pred_sampled, gt_hs)
                    acc_cls_fbg = self.fbg_cls_pixel_acc(pred_sampled, gt_hs)
                    acc_bin_fbg = self.fbg_bin_pixel_acc(pred_sampled, gt_hs)
            if self.cfg.TRAIN.deform_joint_loss:
                # enter here!!!!
                if not is_inference:
                    return loss, acc, edge_loss
                else:
                    return loss, acc, edge_loss, acc_bin_fg, acc_cls_fbg, acc_bin_fbg
            else:
                if not is_inference:
                    return loss, acc
                else:
                    return loss, acc, acc_bin_fg, acc_cls_fbg, acc_bin_fbg


class SegmentationModule(SegmentationModuleBase):
    def __init__(self, net_enc, net_dec, crit, cfg, deep_sup_scale=None, net_fov_res=None):
        super(SegmentationModule, self).__init__()
        self.encoder = net_enc
        self.decoder = net_dec
        self.crit = crit
        self.cfg = cfg
        self.deep_sup_scale = deep_sup_scale
        self.net_fov_res = net_fov_res

    # @torchsnooper.snoop()
    def forward(self, feed_dict, *, segSize=None, F_Xlr_acc_map=False, writer=None, count=None, feed_dict_info=None, feed_batch_count=None):
        # training
        if segSize is None:
            if self.deep_sup_scale is not None: # use deep supervision technique
                (pred, pred_deepsup) = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True))
            elif self.net_fov_res is not None:
                pred = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True), res=self.net_fov_res(feed_dict['img_data']))
            else:
                pred = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True))

            loss = self.crit(pred, feed_dict['seg_label'])
            if self.deep_sup_scale is not None:
                loss_deepsup = self.crit(pred_deepsup, feed_dict['seg_label'])
                loss = loss + loss_deepsup * self.deep_sup_scale

            acc = self.pixel_acc(pred, feed_dict['seg_label'])
            return loss, acc
        # inference
        else:
            if self.net_fov_res is not None:
                pred = self.decoder(self.encoder(feed_dict['img_data'].contiguous(), return_feature_maps=True), segSize=segSize, res=self.net_fov_res(feed_dict['img_data']))
            else:
                pred = self.decoder(self.encoder(feed_dict['img_data'].contiguous(), return_feature_maps=True), segSize=segSize)
            if self.cfg.VAL.write_pred:
                _, pred_print = torch.max(pred, dim=1)
                colors = loadmat('data/color150.mat')['colors']
                pred_color = colorEncode(as_numpy(pred_print.squeeze(0)), colors)
                pred_print = torch.from_numpy(pred_color.astype(np.uint8)).unsqueeze(0).permute(0,3,1,2)
                print('train/pred size: {}'.format(pred_print.shape))
                pred_print = vutils.make_grid(pred_print, normalize=False, scale_each=True)
                writer.add_image('train/pred', pred_print, count)

            if F_Xlr_acc_map:
                loss = self.crit(pred, feed_dict['seg_label'])
                return pred, loss
            else:
                return pred

class ModelBuilder:
    # custom weights initialization
    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
    @staticmethod
    def build_encoder(arch='resnet50', fc_dim=2048, weights='', dilate_rate=4):
        pretrained = True if len(weights) == 0 else False
        arch = arch.lower()
        if arch == 'hrnetv2_nodownsp':
            print('use hrnet!!!!!!!!!!!!!')
            net_encoder = hrnetv2_nodownsp.__dict__['hrnetv2_nodownsp'](pretrained=False)
        elif arch == 'segformer':
            print('use segformer!!!!!!!!!!!!!')
            net_encoder = segformer.__dict__['segformer'](pretrained=False)
        elif arch == 'deeplab':
            print('use deeplab!!!!!!!!!!!!!')
            net_encoder = deeplab.__dict__['deeplab'](pretrained=False)
        else:
            raise Exception('Architecture undefined!')

        # encoders are usually pretrained
        # net_encoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            print(f'Loading weights for net_encoder {weights}')
            net_encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_encoder

    @staticmethod
    def build_decoder(arch='upernet',
                      fc_dim=2048, num_class=150,
                      weights='', use_softmax=False):
        arch = arch.lower()
        if arch == 'c1':
            net_decoder = C1(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        else:
            raise Exception('Architecture undefined!')

        net_decoder.apply(ModelBuilder.weights_init)
        # net_decoder.load_state_dict(
        #     torch.load('/root/autodl-tmp/lvis_Tin_80_80_ours_1_8_9_58pm_mse_dicefocal_kernel45_12w_largerhr/decoder_epoch_120.pth', map_location=lambda storage, loc: storage), strict=False)
        if len(weights) > 0:
            print('Loading weights for net_decoder')
            net_decoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_decoder

    @staticmethod
    def build_net_saliency(cfg=None,
                        weights=''):
        # define saliency network
        # Spatial transformer localization-network
        if cfg.MODEL.track_running_stats:       # True
            if cfg.MODEL.saliency_net == 'fovsimple':
                net_saliency = fov_simple(cfg)

        if len(weights) == 0:
            net_saliency.apply(ModelBuilder.weights_init)
        else:
            print('Loading weights for net_saliency')
            net_saliency.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_saliency

    @staticmethod
    def build_net_compress(cfg=None,
                        weights=''):
        net_compress = CompressNet(cfg)

        if len(weights) == 0:
            net_compress.apply(ModelBuilder.weights_init)
        else:
            print('Loading weights for net_compress')
            net_compress.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_compress
