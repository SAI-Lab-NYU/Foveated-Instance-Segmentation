# Foveated Instance Segmentation
## Environment Setup
```bash
pip install -r requirements.txt
```

## Prepare data
1. Prepare LVIS data (50 classes, sp60000 for train, sp12000 for valid). The data structure should be:
    ```bash
    b_data_train
    ├── data_a_raw
    │   ├── lvis_v1_train
    │   │   └──lvis_v1_train.json
    │   ├── lvis_v1_val
    │   │   └──lvis_v1_val.json
    │   └── coco2017
    │       ├──train2017
    │       ├──val2017
    │       ├──test2017
    │       └──annotation
    ├── data_b_cache
    ├── data_c_cook
    │   └── lvis
    │       ├── train
    │       │   └── sp60000
    │       └── valid
    │           └── sp12000
    ```
    Dataset for COCO is from:
    ```bash
    http://images.cocodataset.org/zips/train2017.zip
    http://images.cocodataset.org/zips/val2017.zip
    http://images.cocodataset.org/zips/test2017.zip
    http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip
    ```
    Dataset for LVIS is from:
    ```bash
    https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip
    https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip
    ```
2. Set up the paths in ```/DynamicFocus/preset.py```. Point ```dpath_data_raw```, ```dpath_data_cache```, ```dpath_data_cook``` to the paths in step 1.

3. Generate data, 
    ```bash
    cd DynamicFocus
    python e_preprocess_scripts/b2_preprocess_lvis.py --task preprocess --dataset_partition train valid --sample_num 60000
    ```

## Run command
### Evaluate the current ckpt on lvis 50 classes data
1. ckpt address: https://drive.google.com/drive/folders/1sxYLCFNCaei7IbXFGQ69lWGE3TSEgraz?usp=sharing
    download lvis_50cls_ckpt.zip and unzip

2. Copy ckpt
    ```bash
    mkdir ckpt
    copy -r lvis_50cls ./ckpt/lvis_50cls
    ```

3. Run evaluation
    ```bash
    CUDA_VISIBLE_DEVICES=0,1 python3 train_deform_semantic.py --gpus 0-1 --cfg config/deform.yaml TRAIN.task_input_size '(80,80)' DIR "./ckpt/lvis_50cls_hrnet" TRAIN.deform_joint_loss True VAL.no_upsample True TRAIN.num_epoch 121 TRAIN.start_epoch 120 TRAIN.eval_per_epoch 1 TRAIN.skip_train_for_eval True VAL.no_upsample True DATASET.dataset_marker_train 'sp60000' DATASET.dataset_marker_valid 'sp12000' MODEL.gaussian_radius 45 TRAIN.saliency_input_size '(80, 80)'
    ```

### Train on lvis 50 classes data
1. Train command
    ```bash
    CUDA_VISIBLE_DEVICES=0,1 python3 train_deform_semantic.py --gpus 0-1 --cfg config/deform.yaml TRAIN.task_input_size '(80,80)' DIR "./ckpt/lvis_50cls_hr_net_train" TRAIN.deform_joint_loss True VAL.no_upsample True TRAIN.num_epoch 150 TRAIN.eval_per_epoch 10 TRAIN.checkpoint_per_epoch 20 TRAIN.skip_train_for_eval False VAL.no_upsample True DATASET.dataset_marker_train 'sp60000' DATASET.dataset_marker_valid 'sp12000' MODEL.gaussian_radius 45 TRAIN.saliency_input_size '(80, 80)'
    ```

### Other models
1. Segformer
    In command, add ```MODEL.arch_encoder 'segformer' MODEL.fc_dim 1024```
2. Deeplab
    In command, add ```MODEL.arch_encoder 'deeplab'```