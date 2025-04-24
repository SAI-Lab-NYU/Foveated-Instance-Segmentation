import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.nn.functional as F
import torchvision.transforms as T
import os
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from .. import preset

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
#original_idx_to_new_idx = {77: 0, 421: 1, 816: 2, 1115: 3, 225: 4, 173: 5, 1052: 6, 982: 7, 1021: 8, 1050: 9, 76: 10, 110: 11, 361: 12, 1097: 13, 631: 14, 422: 15, 378: 16, 687: 17, 1077: 18, 912: 19, 183: 20, 445: 21, 1019: 22, 818: 23, 496: 24, 817: 25, 3: 26, 698: 27, 703: 28, 766: 29, 1123: 30, 19: 31, 1202: 32, 1071: 33, 1042: 34, 694: 35, 296: 36, 569: 37, 1117: 38, 1064: 39, 961: 40, 350: 41, 461: 42, 169: 43, 1043: 44, 392: 45, 36: 46, 90: 47, 181: 48, 139: 49, 150: 50, 1139: 51, 271: 52, 80: 53, 828: 54, 346: 55, 748: 56, 923: 57, 143: 58, 351: 59, 804: 60, 793: 61, 1142: 62, 230: 63, 898: 64, 94: 65, 719: 66, 1037: 67, 713: 68, 1110: 69, 836: 70, 708: 71, 1133: 72, 589: 73, 1008: 74, 692: 75, 118: 76, 96: 77, 1060: 78, 61: 79, 498: 80, 595: 81, 968: 82, 277: 83, 947: 84, 387: 85, 50: 86, 1026: 87, 835: 88, 66: 89, 207: 90, 592: 91, 881: 92, 976: 93, 716: 94, 347: 95, 1190: 96, 154: 97, 1079: 98, 1177: 99}
dpath_data_raw = preset.dpath_data_raw 
dpath_data_raw_coco_train = os.path.join(dpath_data_raw, r'coco2017', r'train2017')
dpath_data_raw_coco_valid = os.path.join(dpath_data_raw, r'coco2017', r'val2017')
dpath_data_raw_coco_test = os.path.join(dpath_data_raw, r'coco2017', r'test2017')

def convert_index(original_index):
    return original_idx_to_new_idx.get(original_index, 19)


class PreprocessDataset(Dataset):
    def __init__(self, data_path, marker, dataset_partition='train', dataset_name='cityscapes', transform=None):
        self.HC = 512 if dataset_name == 'cityscpaes' else 640
        self.WC = 1024 if dataset_name == 'cityscpaes' else 640
        self.K = len(original_idx_to_new_idx)
        self.data_path = data_path
        self.marker = marker
        self.dataset_partition = dataset_partition
        self.transform = transform
        self.dataset_name = dataset_name
        self.coco_path = dpath_data_raw_coco_train if dataset_partition =='train' else dpath_data_raw_coco_valid

        self.dpath_data_cook_data_part_mark = os.path.join(
            self.data_path,
            dataset_name,
            self.dataset_partition,
            self.marker
        )

        self.data_info = []
        if dataset_name == 'cityscapes':
            for entry in os.scandir(self.dpath_data_cook_data_part_mark):
                if entry.name.endswith('.Y.pt') and entry.is_file():
                    fname_Y = entry.name
                    caty, cid, kid, itemkey, fpos, IxHxW_Y = fname_Y.split('.')[0].split('_')
                    IxHxW_X = '3x' + IxHxW_Y[2:]
                    fname_X = f"{caty}_{cid}_{kid}_{itemkey}_{fpos}_{IxHxW_X}.uint8.X.pt"

                    fpath_Y = os.path.join(self.dpath_data_cook_data_part_mark, fname_Y)
                    fpath_X = os.path.join(self.dpath_data_cook_data_part_mark, fname_X)
                    idx_H, idx_W = map(int, fpos.split('x'))
                    Y_cls_s = convert_index(int(kid[1:]))

                    self.data_info.append({
                        'fpath_Y': fpath_Y,
                        'fpath_X': fpath_X,
                        'idx_H': idx_H,
                        'idx_W': idx_W,
                        'Y_cls_s': Y_cls_s
                    })
        elif dataset_name == 'lvis':
            for entry in os.scandir(self.dpath_data_cook_data_part_mark):
                if entry.name.endswith('.Y.pt') and entry.is_file():
                    fname_Y = entry.name
                    caty, cid, kid, aid, imgid, fpos, paddings, IxHxW = fname_Y.split('.')[0].split('_')
                    Y_cls_s = int(kid[1:])
                    pad_left, pad_right, pad_top, pad_bottom = [int(num) for num in paddings.split('x')]
                    idx_H, idx_W = [int(num) for num in fpos.split('x')]
                    fpath_Y = os.path.join(self.dpath_data_cook_data_part_mark, fname_Y)

                    fname_img = f"{imgid}.jpg"
                    fpath_img = os.path.join(self.coco_path, fname_img)

                    if not os.path.exists(fpath_img):
                        for dpath_coco in [dpath_data_raw_coco_train, dpath_data_raw_coco_valid, dpath_data_raw_coco_test]:
                            fpath_img = os.path.join(dpath_coco, fname_img)
                            if os.path.exists(fpath_img):
                                break

                    self.data_info.append({
                        'fpath_Y': fpath_Y,
                        'fpath_X': fpath_img,
                        'idx_H': idx_H,
                        'idx_W': idx_W,
                        'Y_cls_s': Y_cls_s,
                        'pad_left': pad_left,
                        'pad_right': pad_right,
                        'pad_top': pad_top,
                        'pad_bottom': pad_bottom,
                    })

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        if self.dataset_name == 'cityscapes':
            info = self.data_info[idx]
            Y_1xHxW = torch.load(info['fpath_Y'], weights_only=True).float()
            X_3xHxW = torch.load(info['fpath_X'], weights_only=True).float() / 255.0
            F_2 = torch.tensor([info['idx_H'] / self.HC, info['idx_W'] / self.WC], dtype=torch.float32)
            Y_cls_1 = torch.tensor([info['Y_cls_s']], dtype=torch.int64)

            if self.transform:
                X_3xHxW = self.transform(X_3xHxW)

            return X_3xHxW, F_2, Y_1xHxW, Y_cls_1
        else:
            info = self.data_info[idx]
            Y_1xHxW = torch.load(info['fpath_Y'], weights_only=True).float()
            image = Image.open(info['fpath_X']).convert('RGBA')
            transform_to_tensor = T.ToTensor()
            X_4xHxW = transform_to_tensor(image).to(dtype=torch.float32)
            Y_seg_1xHPxWP = F.pad(Y_1xHxW, (info['pad_left'], info['pad_right'], info['pad_top'], info['pad_bottom'])).to(dtype=torch.float32)
            X_4xHPxWP = F.pad(X_4xHxW, (info['pad_left'], info['pad_right'], info['pad_top'], info['pad_bottom']))
            F_2 = torch.Tensor([info['idx_H'] / self.HC, info['idx_W'] / self.WC]).to(dtype=torch.float32)

            Y_cls_1 = torch.Tensor([info['Y_cls_s']]).to(dtype=torch.int64)

            return X_4xHPxWP, F_2, Y_seg_1xHPxWP, Y_cls_1



def init_distributed_mode(world_size, rank, backend='nccl'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def main(rank, world_size):
    init_distributed_mode(world_size=world_size, rank=rank)
    data_path = '/home/lwx/b_data_train/data_c_cook'
    marker = 'sp500'

    dataset = PreprocessDataset(data_path=data_path, marker=marker, dataset_name='lvis',dataset_partition='train')
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=10, sampler=sampler, num_workers = 2)

    for epoch in range(5):
        sampler.set_epoch(epoch)
        if rank == 0: 
            dataloader = tqdm(dataloader, desc=f"Epoch {epoch}, Rank {rank}")

        for images, _, targets, _ in dataloader:
            images, targets = images.to(rank), targets.to(rank)

if __name__ == '__main__':

    world_size = 2

    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)