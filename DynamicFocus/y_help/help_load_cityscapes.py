import os

from tqdm import tqdm

import torch
import torchvision.transforms as T
from PIL import Image

import os
import platform

# 检查当前操作系统类型
if platform.system() == 'Windows':
    pc_name = os.environ.get('COMPUTERNAME')
else:
    pc_name = os.environ.get('HOSTNAME')

if pc_name == 'XPS':
    HS = 256
    WS = 512
    path_data_cook_cityscape_part_canvsize = r'D:\b_data_train\data_c_cook\cityscape\train\256x512'

elif pc_name == 'sn4622121202':
    HS = 256
    WS = 512
    path_data_cook_cityscape_part_canvsize = r'/home/hongyiz/DriverD/b_data_train/data_c_cook/cityscape/train/256x512'

device_target_global = torch.device(f'cuda:0') if torch.cuda.device_count() >= 1 else torch.device('cpu')


def load_image(path: str, mode='RGB'):
    image = Image.open(path).convert(mode)
    transform_to_tensor = T.ToTensor()
    view_rgb_3xHxW = transform_to_tensor(image).to(dtype=torch.float32, device=target_device)
    return view_rgb_3xHxW


def save_image(image_tensor: torch.Tensor, path: str):
    transform_to_pil = T.ToPILImage()
    result_image = transform_to_pil(image_tensor)
    result_image.save(path)


def load_tensor(path: str, device=None):
    tensor = torch.load(path, weights_only=True)
    tensor_to_device = tensor.to(device)
    return tensor_to_device


def save_tensor(tensor: torch.Tensor, path: str):
    tensor_to_save = tensor.detach().cpu()
    torch.save(tensor_to_save, path)
    # print(f"Tensor save {path}.")


def get_all_label2namekeys():
    label2namekeys = {}

    for fname in os.listdir(path_data_cook_cityscape_part_canvsize):
        if fname.endswith('.X.pt'):
            name = fname.split('.')[0]
            label = name.split('_')[0]
            if label not in label2namekeys:
                label2namekeys[label] = []
            label2namekeys[label].append(name)

    return label2namekeys


def load_a_sample(namekey, device=device_target_global):
    fpath_X = os.path.join(path_data_cook_cityscape_part_canvsize, f'{namekey}.3x{HS}x{WS}.uint8.X.pt')
    fpath_Y = os.path.join(path_data_cook_cityscape_part_canvsize, f'{namekey}.1x{HS}x{WS}.uint8.Y.pt')
    idx_H, idx_W = map(float, namekey.split('_')[-1].split('x'))

    X_rgb_1x3xHxW = load_tensor(fpath_X).to(dtype=torch.float32, device=device).unsqueeze(0) / 255.0
    X_focus_1x2 = torch.tensor([idx_H / HS, idx_W / WS], dtype=torch.float32, device=device_target_global).unsqueeze(0)
    Y_1xHxW = load_tensor(fpath_Y).to(dtype=torch.int64, device=device).unsqueeze(0)
    return X_rgb_1x3xHxW, X_focus_1x2, Y_1xHxW


def load_all_samples(device=device_target_global, namekeys_all=[]):
    X_rgb_1x3xHxW_s, X_focus_1x2_s, Y_1xHxW_s = [], [], []
    label2namekeys = get_all_label2namekeys()
    for label, namekeys_sub in tqdm(label2namekeys.items()):
        for namekey in namekeys_sub:
            X_rgb_1x3xHxW, X_focus_1x2, Y_1xHxW = load_a_sample(namekey, device=device)
            X_rgb_1x3xHxW_s.append(X_rgb_1x3xHxW)
            X_focus_1x2_s.append(X_focus_1x2)
            Y_1xHxW_s.append(Y_1xHxW)
            namekeys_all.append(namekey)
    X_rgb_Bx3xHxW = torch.cat(X_rgb_1x3xHxW_s, dim=0)
    X_focus_Bx2 = torch.cat(X_focus_1x2_s, dim=0)
    Y_Bx1xHxW = torch.cat(Y_1xHxW_s, dim=0)

    print()
    print(f"X_rgb_Bx3xHxW:{X_rgb_Bx3xHxW.shape}")
    print(f"X_focus_Bx2:{X_focus_Bx2.shape}")
    print(f"Y_Bx1xHxW:{Y_Bx1xHxW.shape}")
    print(flush=True)

    return X_rgb_Bx3xHxW, X_focus_Bx2, Y_Bx1xHxW


X_rgb_Bx3xHxW, X_focus_Bx2, Y_Bx1xHxW = load_all_samples()
if __name__ == '__main__':
    pass
    X_rgb_Bx3xHxW, X_focus_Bx2, Y_Bx1xHxW = load_all_samples()
