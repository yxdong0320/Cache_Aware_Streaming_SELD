import os, shutil
import numpy as np
import math
import lmdb
from tqdm import tqdm
from datum_pb2 import SimpleDatum

def npy_to_lmdb(npy_data_dir, npy_label_dir, lmdb_out_dir, segment_length_s
    #, visual_tools
      ):

    lmdb_map_size = 1099511627776

    if os.path.exists(lmdb_out_dir):
        shutil.rmtree(lmdb_out_dir)
    os.makedirs(lmdb_out_dir)
    
    env = lmdb.open(lmdb_out_dir, map_size=lmdb_map_size)
    txn = env.begin(write=True)
    lmdb_key_name  = lmdb_out_dir + "/keys.txt"
    total_key_file = open(lmdb_key_name, 'w')

    # segment_length_s = 10
    segment_label_frame_num = int(segment_length_s / 0.1)  # 100
    segment_data_frame_num = segment_label_frame_num * 5  # 500

    Num_count = 0
    for filename in tqdm(os.listdir(npy_data_dir)):
        # if 'fold4' in filename:
        data = np.load(os.path.join(npy_data_dir, filename))
        label = np.load(os.path.join(npy_label_dir, filename))
        label_frame_num = min(data.shape[0]//5, label.shape[0])
        data_frame_num = label_frame_num * 5

        segment_num = math.ceil(data_frame_num / segment_data_frame_num)
        for seg_id in range(segment_num):
            segment_data = data[seg_id*segment_data_frame_num:(seg_id+1)*segment_data_frame_num]
            segment_label = label[seg_id*segment_label_frame_num:(seg_id+1)*segment_label_frame_num]
            wav_name = filename.split('.')[0] + '_seg_{}_{}'.format(segment_num, seg_id)

            datum = SimpleDatum()
            # _, _, visual_data = visual_tools.load_file.load_keypoint_feature(data_cat=data_cat, file_name=_filename)
            datum.data = segment_data.astype(np.float32).tobytes()
            datum.label = segment_label.astype(np.float32).tobytes()
            datum.data_dim = segment_data.reshape(segment_data.shape[0], -1).shape[-1]
            datum.label_dim = segment_label.reshape(segment_label.shape[0], -1).shape[-1]
            datum.wave_name = wav_name.encode()
            txn.put(wav_name.encode(), datum.SerializeToString())
            total_key_file.write('{}\n'.format(wav_name))
            total_key_file.flush()

            Num_count += 1
            if (Num_count % 1000) == 0:
                print("save the %d sample" % Num_count)
                txn.commit()
                txn = env.begin(write=True)
    
    print("save the %d sample" % Num_count)
    txn.commit()
    env.close()
    total_key_file.close()

def debug_read_lmdb(lmdb_dir):
    env = lmdb.open(lmdb_dir, readonly=True, readahead=True, lock=False)
    txn = env.begin()
    with open(os.path.join(lmdb_dir, 'keys.txt'), 'r') as f:
        keys = f.readlines()
    with txn.cursor() as cursor:
        for key in keys:
            k = key.strip().encode()
            cursor.set_key(k)
            datum=SimpleDatum()
            datum.ParseFromString(cursor.value())
            data = np.fromstring(datum.data, dtype=np.float32).reshape(-1, datum.data_dim)
            label = np.fromstring(datum.label, dtype=np.float32).reshape(-1, datum.label_dim)
            wav_name = datum.wave_name.decode()
            print(wav_name)
            print('data:', data.shape)
            print('label:', label.shape)

if __name__ == "__main__":
    npy_data_dir = '/disk6/yxdong/Dcase2023/synth_real_data/combine_synth_real/feat_label/foa_dev'
    # 音频特征目录下的子目录/foa_dev,上一步会自动生成
    npy_label_dir = '/disk6/yxdong/Dcase2023/synth_real_data/combine_synth_real/feat_label/foa_dev_label'
    # 子目录/foa_dev_label
    lmdb_out_dir = '/disk6/yxdong/Dcase2023/synth_real_data/combine_synth_real/lmdb_foa_dev_data_label_len1s'
    # LMDB文件存放目录
    segment_length_s=1
    npy_to_lmdb(npy_data_dir, npy_label_dir, lmdb_out_dir,segment_length_s)
    debug_read_lmdb(lmdb_out_dir)

