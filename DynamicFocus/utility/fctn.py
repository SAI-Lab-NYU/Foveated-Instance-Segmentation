import itertools
import pickle

import pandas as pd

import datetime
import json

import torch
import torchvision.transforms as T
from PIL import Image


def get_args_kwargs(*args, **kwargs):
    return args, kwargs


def get_dctns_cols_by_df(df: pd.DataFrame):
    cols = df.columns.values.tolist()
    dctns = [{k: v for k, v in zip(cols, row)} for row in df.values.tolist()]
    return dctns, cols


def get_df_by_dctns_cols(dctns, cols):
    df = pd.DataFrame(dctns, columns=cols)
    return df


def save_jsonl(data, filename):
    """
    将数据保存为JSONL格式的文件。

    :param data: 要保存的数据，列表中的每个元素都应该是可序列化为JSON的对象。
    :param filename: 保存文件的名称。
    """
    with open(filename, 'w', encoding='utf-8') as file:
        for entry in data:
            json_line = json.dumps(entry, ensure_ascii=False)  # 将对象转换为JSON字符串
            file.write(json_line + '\n')  # 写入文件，每个对象后换行


def read_jsonl(filename):
    """
    从JSONL格式的文件中读取数据。

    :param filename: JSONL文件的名称。
    :return: 包含文件中所有JSON对象的列表。
    """
    data = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))  # 读取每行并转换为Python对象
    return data


def save_json(data, filename):
    """
    将数据保存为JSON格式的文件。

    :param data: 要保存的数据，应该是可序列化为JSON的对象。
    :param filename: 保存文件的名称。
    """
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)  # 将对象转换为JSON并保存到文件


def read_json(filename):
    """
    从JSON格式的文件中加载数据。

    :param filename: JSON文件的名称。
    :return: 文件中的JSON对象。
    """
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)  # 读取并转换整个文件的JSON内容


def save_text(data, filename):
    """
    将字符串数据保存到文本文件中。

    :param data: 要保存的字符串数据。
    :param filename: 保存文件的名称。
    """
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(data)  # 将字符串写入文件


def read_text(filename):
    """
    从文本文件中读取字符串数据。

    :param filename: 文本文件的名称。
    :return: 文件内容的字符串。
    """
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()  # 读取整个文件内容并返回


def date2datetime(dt: datetime.date):
    return datetime.datetime.combine(dt, datetime.time())


def load_image(path: str, mode='RGB'):
    image = Image.open(path).convert(mode)
    transform_to_tensor = T.ToTensor()
    view_rgb_3xHxW = transform_to_tensor(image).to(dtype=torch.float32)
    return view_rgb_3xHxW


def save_image(image_tensor: torch.Tensor, path: str):
    transform_to_pil = T.ToPILImage()
    result_image = transform_to_pil(image_tensor)
    result_image.save(path)


def load_tensor(path: str):
    tensor = torch.load(path, weights_only=True)
    return tensor


def save_tensor(tensor: torch.Tensor, path: str):
    tensor_to_save = tensor.detach().cpu()
    torch.save(tensor_to_save, path)
    # print(f"Tensor save {path}.")


def read_pickle(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


def save_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)


if __name__ == '__main__':
    pass
