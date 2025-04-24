import gc
import subprocess
import time
from typing import Union
import datetime

import pandas as pd
import torch
import numpy as np
from torch import nn


def get_gpu_meminfo() -> dict:
    try:
        # 运行 nvidia-smi 并获取 GPU 内存信息
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,nounits,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )

        # 初始化 GPU 内存字典
        gpu_memory_info = {}

        # 解析 nvidia-smi 的输出
        lines = result.stdout.strip().split("\n")
        for idx, line in enumerate(lines):
            total_memory, used_memory = map(int, line.split(","))
            free_memory = total_memory - used_memory

            # 将每个 GPU 的可用内存存储到字典中
            gpu_memory_info[idx] = {
                "total_memory": total_memory,
                "used_memory": used_memory,
                "free_memory": free_memory
            }

    except subprocess.CalledProcessError as e:
        print(f"Failed to run nvidia-smi: {e.stderr}")
        return None

    return gpu_memory_info


def try_gpu(memory_threshold=1024 * 6, block=True, retry_interval=60, gpu_index=None):
    """
    memory_threshold: in MB
    """

    if gpu_index is not None:
        return torch.device(f'cuda:{gpu_index}')
    else:
        while True:
            gpu_meminfo = get_gpu_meminfo()

            gpuid_best = sorted(list(gpu_meminfo.keys()), key=lambda gid: gpu_meminfo[gid]['free_memory'], reverse=True)[0]

            max_freememory = gpu_meminfo[gpuid_best]['free_memory']

            if max_freememory > memory_threshold:
                print(f"Select GPU {gpuid_best} free memory: {max_freememory} MB used/total memory = {gpu_meminfo[gpuid_best]['used_memory']}/{gpu_meminfo[gpuid_best]['total_memory']} MB")

                return torch.device(f'cuda:{gpuid_best}')
            else:
                sent = f"No suitable GPU found. current max_freememory = {max_freememory} < {memory_threshold} = memory_threshold"

                if block:
                    print(sent + f" : retrying in {retry_interval} seconds...")
                    time.sleep(retry_interval)
                else:
                    raise Exception(sent)


def try_cpu():
    return torch.device('cpu')


def init_weights_zero(m):
    '''
        Usage:
            d_model = Model()
            d_model.apply(weight_init)
        '''
    if isinstance(m, nn.Conv1d):
        nn.init.zeros_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        nn.init.zeros_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        nn.init.zeros_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        nn.init.zeros_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.zeros_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        nn.init.zeros_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.zeros_(m.weight.data)
        nn.init.zeros_(m.bias.data)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.zeros_(m.weight.data)
        nn.init.zeros_(m.bias.data)
    elif isinstance(m, nn.BatchNorm3d):
        nn.init.zeros_(m.weight.data)
        nn.init.zeros_(m.bias.data)
    elif isinstance(m, nn.Linear):
        nn.init.zeros_(m.weight.data)
        nn.init.zeros_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            nn.init.zeros_(param.data)


def init_weights_random(m):
    '''
    Usage:
        d_model = Model()
        d_model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        nn.init.normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        nn.init.normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)


def calc_tensor_memsize(tensor, unit='MB'):
    # 获取张量的总元素数目以及每个元素占用的字节数
    num_elements = tensor.numel()
    element_size = tensor.element_size()

    # 计算张量总大小（以字节为单位）
    total_size_in_bytes = num_elements * element_size

    # 单位换算
    if unit == 'KB':
        return total_size_in_bytes / 1024
    elif unit == 'MB':
        return total_size_in_bytes / (1024 ** 2)
    elif unit == 'GB':
        return total_size_in_bytes / (1024 ** 3)
    else:
        raise ValueError("Invalid unit. Please choose from 'KB', 'MB', or 'GB'.")


def calc_model_memsize(model, unit='MB', show=True, label='the model'):
    total_size = 0
    for param in model.parameters():
        total_size += calc_tensor_memsize(param, unit)

    if show:
        print(f"Total memory size of {label} in {unit}: {total_size:.4f} {unit}")
    return total_size


def show_model_info(model, show_details=True):
    table = []
    total_ele = 0
    total_mem = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        tsr = parameter.data

        nele = tsr.nelement()
        sele = tsr.element_size()

        msize = nele * sele

        table.append([name, nele, round(msize / 1024, 2)])
        total_ele += nele
        total_mem += msize
    df = pd.DataFrame(table, columns=["Modules", "Parameters", "Mem (KB)"])

    if show_details:
        with pd.option_context('display.max_rows', None,
                               'display.max_columns', None,
                               'display.precision', 3,
                               ):
            print(df)

    total_mem = round(total_mem / 1024 ** 2, 4)

    print(f"Total Trainable : {total_ele} ele , {total_mem} MB\n")
    return total_ele, total_mem


def byte2MB(bt):
    return round(bt / (1024 ** 2), 3)


class RAM:
    data = {}
    records_reserved = 0
    records_allocate = 0

    def __init__(self):
        pass

    def __setattr__(self, name, value):
        RAM.data[name] = value

    def __getattr__(self, name):
        return RAM.data[name]

    def __delattr__(self, name):
        if name in RAM.data:
            del RAM.data[name]

    def __setitem__(self, key, value):
        RAM.data[key] = value

    def __getitem__(self, item):
        return RAM.data[item]

    def __delitem__(self, key):
        if key in RAM.data:
            del RAM.data[key]

    def keys(self):
        return RAM.data.keys()

    def gc(self):
        gc.collect()
        torch.cuda.empty_cache()

    def delete_all(self, show=False):
        for name in list(RAM.data.keys()):
            if show:
                print(f"del {name}")
            del RAM.data[name]

    def show_cuda_info(self):
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r - a

        print('Cuda Info')
        print(f'Total     \t{byte2MB(t)} MB')
        print(f'Reserved  \t{byte2MB(r)} MB')
        print(f'Allocated \t{byte2MB(a)} MB')
        print(f'Free      \t{byte2MB(f)} MB')


if __name__ == '__main__':
    pass

    """
    ram = RAM()

    print("---- Testing __setattr__ and __getattr__ ----")
    ram.tensor_a = torch.randn(1000, 1000, device='cuda')
    print("Tensor a stored in RAM:")
    print(ram.tensor_a)

    print("\n---- Testing __setitem__ and __getitem__ ----")
    ram['tensor_b'] = torch.randn(2000, 2000, device='cuda')
    print("Tensor b stored in RAM:")
    print(ram['tensor_b'])

    print("\n---- Testing keys ----")
    print("Current keys in RAM:", list(ram.keys()))

    print("\n---- Testing __delattr__ (deleting tensor_a) ----")
    del ram.tensor_a
    print("tensor_a deleted.")
    print("Current keys in RAM:", list(ram.keys()))

    print("\n---- Testing __delitem__ (deleting tensor_b) ----")
    del ram['tensor_b']
    print("tensor_b deleted.")
    print("Current keys in RAM:", list(ram.keys()))

    print("\n---- Testing delete_all (adding and clearing tensors) ----")
    ram.tensor_c = torch.randn(1500, 1500, device='cuda')
    ram['tensor_d'] = torch.randn(3000, 3000, device='cuda')
    print("Current keys in RAM before delete_all:", list(ram.keys()))
    ram.delete_all()
    print("All tensors deleted.")
    print("Current keys in RAM after delete_all:", list(ram.keys()))

    print("\n---- Testing show_cuda_info ----")
    ram.show_cuda_info()

    print("\n---- Testing gc (Garbage Collection and CUDA cache clearing) ----")
    ram.gc()
    print("Garbage collection and CUDA memory cache cleared.")
    ram.show_cuda_info()
    """

    try_gpu()
