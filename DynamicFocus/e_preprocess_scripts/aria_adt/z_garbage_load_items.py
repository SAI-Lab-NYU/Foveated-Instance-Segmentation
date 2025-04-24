"""
file_path = os.path.join(dpath_cache_aria_adt, skey, f"ADT_{skey}_preview_rgb.mp4")
dpath_img = os.path.join(dpath_cache_aria_adt, skey, 'img_pth')
os.makedirs(dpath_img, exist_ok=True)

# 打开视频文件
cap = cv2.VideoCapture(file_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频的总帧数

for i in tqdm(range(total_frames), desc="Reading video frames", total=total_frames):
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 将帧从 HxWx3 转换为 3xHxW，并转换为 PyTorch 张量
    frame_tensor = torch.from_numpy(frame)  # 归一化到 [0, 1]

    tensor_img_rotated_CxHxW = frame_tensor.permute(2, 0, 1)

    save_tensor(tensor_img_rotated_CxHxW, os.path.join(dpath_img, f"{timestamp_ns_s[i]}.{str_tensor_shape(tensor_img_rotated_CxHxW)}.pth"))
    save_image(tensor_img_rotated_CxHxW, os.path.join(dpath_img, f"{timestamp_ns_s[i]}.{str_tensor_shape(tensor_img_rotated_CxHxW)}.png"))

cap.release()

#
    #
    # def pixel_rotate_90(u, v, HW_size=1408):
    #     return u, HW_size - 1 - v



    #
    # aa.download_a_sequence()

    # sequence_path = r'/mnt/d/b_data_train/data_b_cache/aria_adt/Apartment_release_clean_seq131_M1292'
    # sequence_path = r'D:\b_data_train\data_b_cache\aria_adt\Apartment_release_clean_seq131_M1292'
    # paths_provider = AriaDigitalTwinDataPathsProvider(sequence_path)
    # data_paths = paths_provider.get_datapaths()
    # print("loading ground truth data...")
    # gt_provider = AriaDigitalTwinDataProvider(data_paths)
    # stream_id =
    #
    # select_timestamps_ns = 13860350878462
    # segmentation_with_dt = gt_provider.get_segmentation_image_by_timestamp_ns(select_timestamps_ns, stream_id)

    # aa.download_a_sequence()
    timestamp_ns_s = aa.get_timestamp_ns()
    gt_provider = aa.get_gt_provider()
    # res = aa.gen_seg_pth(skey=sample_skey, gt_provider=gt_provider, timestamp_ns_s=timestamp_ns_s)
    #
    # res = aa.gen_img_pth(skey=sample_skey, gt_provider=gt_provider, timestamp_ns_s=timestamp_ns_s)

    # dpath_seq = os.path.join(dpath_cache_aria_adt, sample_skey)
    #
    # src_file = os.path.join(dpath_seq, f'ADT_{sample_skey}_main_recording.vrs')
    # copy_destination = os.path.join(dpath_seq, f'video.vrs')
    # shutil.copy(src_file, copy_destination)

    # res = aa.gen_gaze_obj_json(skey=sample_skey, gt_provider=gt_provider, timestamp_ns_s=timestamp_ns_s)

    res = aa.merge_img_seg_gaze_obj(skey=sample_skey)

    # res2 = aa.get_timestamp_ns(sample_skey, 'eyegaze.csv', 'tracking_timestamp_us')

    # # 进入 wsl
    # >>> wsl
    #
    # # 在 wsl 下开启venv
    # >>> source /mnt/d/a_python_venv/DynamicFocusWSL/venv/bin/activate
    # python /e_preprocess_scripts/aria_adt/load_and_make_cache.py
    # pprint(aa.efminfo)
    #
    # print(sum(aa.byte_s))
    #
    # filename = "AriaDigitalTwin_2_0_ATEK_efm_Apartment_release_clean_seq134_M1292_shards-0007.tar"
    # foldername = filename.split('.')[0]
    # url = "https://scontent.xx.fbcdn.net/m1/v/t6/An9bEfFev4p6dVyyJSscawM4hlkITFs1crdtKWPuHD5cOqOsNtGsxBMExGz3iSh2mEJ6GTfFJb1bqE5lhCAafUA2hHqN4mFa-i4AYYoWkqEOYyIzw3nhzdqfCZbNCvxF9HNwUg0oPi9XYXXogfCbZ7CGy58gGceVBY4APnrKZjDYdVo.tar/AriaDigitalTwin_2_0_ATEK_efm_Apartment_release_clean_seq134_M1292_shards-0007.tar?ccb=10-5&oh=00_AYD4Y9lphYNTIkF2KDUPxTClpy8CfNWiEUwFSkOuMus0Ag&oe=67A06066&_nc_sid=c228f2"
    # fpath_filename = os.path.join(dpath_cache_aria_adt, filename)
    # fpath_foldername = os.path.join(dpath_cache_aria_adt, foldername)
    # download_file(url, fpath_filename)
    # extract_tar_to_folder(fpath_filename, fpath_foldername)
    # os.remove(fpath_filename)
    # time.sleep(2)
    # classify_files(fpath_foldername)

    # print_pth_files_info(r'D:\b_data_train\data_b_cache\aria_adt\AriaDigitalTwin_2_0_ATEK_efm_Apartment_release_clean_seq134_M1292_shards-0007\AriaDigitalTwin_Apartment_release_clean_seq134_M1292_AtekDataSample_000056')

"""
