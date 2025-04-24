import os
import platform

# 定义所有的机器名称
PC_NAME_server_H100 = os.environ.get('HOSTNAME')
PC_NAME_server_4090 = os.environ.get('HOSTNAME')
PC_NAME_harry_local = 'XPS'

# 获取当前进程的机器名称, 获取 Windows 的环境变量 COMPUTERNAME
pc_name = os.environ.get('COMPUTERNAME') if platform.system() == 'Windows' else os.environ.get('HOSTNAME')

# 定义短名称，用来标记训练model的机器
PC_name2shortname = {PC_NAME_server_4090: '4090', PC_NAME_server_H100: 'H100', PC_NAME_harry_local: 'XPS'}
pc_shortname = PC_name2shortname[pc_name]

# 定义 records folder 的名称，用来互传文件，以及查看
dpath_training_records_harry_local = r'C:\Users\harry\Dropbox\AGI\CS-GY-997X\DynamicFocus\a_records_train'
dpath_training_records_server_4090_or_H100 = r'/home/xth/a_records_train'

# 对于每个机器，定义对应的路径
dpath_training_records = None
dpath_data_raw = ''

if pc_name == PC_NAME_harry_local:
    dpath_training_records = dpath_training_records_harry_local

    dpath_data_raw = r'D:\b_data_train\data_a_raw'
    dpath_data_cache = r'D:\b_data_train\data_b_cache'
    dpath_data_cook = r'D:\b_data_train\data_c_cook'



elif pc_name == PC_NAME_server_H100 or pc_name == PC_NAME_server_4090:
    dpath_training_records = dpath_training_records_server_4090_or_H100

    dpath_data_raw = r'/home/xth/b_data_train/data_a_raw'
    dpath_data_cache = r'/home/xth/b_data_train/data_b_cache'
    dpath_data_cook = r'/home/xth/b_data_train/data_c_cook'
else:
    # 如果有其他机器，在这里新增路径
    pass

# 初始化 recordes folder
os.makedirs(dpath_training_records, exist_ok=True)

# 定义 raw 数据集的路径

# cityscape 数据集
dpath_data_raw_cityscape_X = os.path.join(dpath_data_raw, r'leftImg8bit_trainvaltest', r'leftImg8bit')
dpath_data_raw_cityscape_Y = os.path.join(dpath_data_raw, r'gtFine_trainvaltest', r'gtFine')

# lvis 数据集
fpath_data_raw_lvis_train = os.path.join(dpath_data_raw, r'lvis_v1_train', r'lvis_v1_train.json')
fpath_data_raw_lvis_valid = os.path.join(dpath_data_raw, r'lvis_v1_val', r'lvis_v1_val.json')

# coco 数据集，注：lvis 会用coco的图片
dpath_data_raw_coco_train = os.path.join(dpath_data_raw, r'coco2017', r'train2017')
dpath_data_raw_coco_valid = os.path.join(dpath_data_raw, r'coco2017', r'val2017')
dpath_data_raw_coco_test = os.path.join(dpath_data_raw, r'coco2017', r'test2017')
