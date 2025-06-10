import argparse
import os
import numpy as np
import cls_feature_class
import parameters
import logging
import torch
import yaml
import pdb
import time
from tqdm import tqdm
from utils.cls_tools.cls_compute_seld_results import ComputeSELDResults
from models.resnet_conformer_audio import ResnetConformer_sed_doa_nopool
from utils.sed_doa import SedDoaResult, process_foa_input_sed_doa, SedDoaLoss, SedDoaResult_Streaming_Inf
# from utils.cls_tools.cls_compute_seld_results import ComputeSELDResults
from utils.write_csv import write_output_format_file
import joblib
import librosa

def read_wav_in_sliding_windows(wav_file, segment_length_sec=10, hop_length_sec=5):
    # 使用librosa加载音频，保持多通道音频
    audio, sr = librosa.load(wav_file, sr=None, mono=False)  # mono=False 保持原始通道数
    
    # 计算每个段的样本数
    segment_length_samples = segment_length_sec * sr
    hop_length_samples = int(hop_length_sec * sr)
    
    # 获取音频的总长度
    audio_length_samples = audio.shape[-1]  # audio.shape[1] 是样本数（多通道时，维度为 [num_channels, num_samples]）
    
    # 从0秒开始，以段移的长度滑动
    for start_sample in range(0, audio_length_samples, hop_length_samples):

        end_sample = start_sample + segment_length_samples
        
        # 如果结束位置超出音频总长度，调整为音频的最后位置
        if end_sample > audio_length_samples:
            end_sample = audio_length_samples
        
        audio_segment = audio[:, start_sample:end_sample]  # 保留所有通道数据
        
        # 返回当前段的音频数据和时间信息
        start_time = start_sample / sr
        end_time = end_sample / sr
        
        # print(f"读取音频段: {start_time:.2f}s - {end_time:.2f}s")
        
        yield audio_segment, start_time, end_time, audio_length_samples

        if end_sample == audio_length_samples:
            break


def main(args):
    # 设置log
    log_output_folder = os.path.dirname(args['result']['log_output_path'])
    os.makedirs(log_output_folder, exist_ok=True)
    logging.basicConfig(filename=args['result']['log_output_path'], 
                        filemode='w', level=logging.INFO, 
                        format='%(levelname)s: %(asctime)s: %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
    logger = logging.getLogger(__name__)
    logger.info(args)

    model = ResnetConformer_sed_doa_nopool(in_channel=args['model']['in_channel'], 
                                           in_dim=args['model']['in_dim'], out_dim=args['model']['out_dim'])
    result_class = SedDoaResult_Streaming_Inf
    data_process_fn = process_foa_input_sed_doa
    normalized_features_wts_file = args['data']['norm_file']
    spec_scaler = joblib.load(normalized_features_wts_file)
    wav_path = args['data']['wav_path']

    window_length = args['model']['window_length']  # 2
    segment_length = args['model']['segment_length']  # 10
    output_num_frame = args['model']['output_num_frame'] # 10
    window_len = int(window_length) * output_num_frame # 20
    past_len = int((segment_length - window_length) * output_num_frame) # 80
    sr = args['model']['sr']

    params = parameters.get_params()
    dev_feat_cls = cls_feature_class.FeatureClass(params)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)
    logger.info(model)

    if args['model']['pre-train']:
        model.load_state_dict(torch.load(args['model']['pre-train_model']))

    model.eval()
    test_result = result_class()

    for file in tqdm(os.listdir(wav_path)):
        if use_cuda:
            torch.cuda.empty_cache()
        file_path = os.path.join(wav_path, file)
        file_name = os.path.splitext(os.path.basename(file_path))[0]

        result_array = None
        for audio_segment, start_time, end_time, audio_len_samp in read_wav_in_sliding_windows(file_path, 
                                                                            segment_length_sec=segment_length, 
                                                                            hop_length_sec=window_length):
            # 获取当前段的真实长度
            current_segment_length = audio_segment.shape[-1] / sr  # 当前段的长度（单位：秒）
            
            if current_segment_length < 0.1:
                # print(f"音频段时间不足0.1秒，丢弃: {start_time:.2f}s - {end_time:.2f}s")
                continue  # 跳过当前音频段                
            # 提取音频特征
            # print(f"读取音频段: {start_time:.2f}s - {end_time:.2f}s")
            data = dev_feat_cls.extract_audio_feature(audio_segment)
            # 特征规整
            data = spec_scaler.transform(data)
            data = data_process_fn(data)
            data = data.astype(np.float32)
            data = torch.from_numpy(data).to(device) # (7, 500, 64)
            data = data.unsqueeze(0) # (1, 7, 500, 64)
            # pdb.set_trace()
            with torch.no_grad():
                # 得到当前段的模型输出
                output = model(data) # torch.Size([1, 100, 52])

            if start_time == 0:
                # 首段：保留前overlap_len帧
                result_array = output
            elif end_time >= audio_len_samp/sr:  # 最后一段
                output_now = output[:, past_len:, :]
                result_array = torch.cat([result_array, output_now], dim=1)            
            else:
                # 中间段：保留past_len到overlap_len之间的帧
                output_now = output[:, past_len:, :]
                result_array = torch.cat([result_array, output_now], dim=1)            
        
        test_result.add_items(file_name, result_array)
        del result_array

    output_dict = test_result.get_result()

    dcase_output_val_dir = args['result']['dcase_output_dir']
    os.makedirs(dcase_output_val_dir, exist_ok=True)
    for csv_name, perfile_out_dict in output_dict.items():
        output_file = os.path.join(dcase_output_val_dir, '{}.csv'.format(csv_name))
        write_output_format_file(output_file, perfile_out_dict)

    score_obj = ComputeSELDResults(ref_files_folder=args['data']['ref_files_dir'])
    val_ER, val_F, val_LE, val_LR, val_seld_scr, classwise_val_scr = score_obj.get_SELD_Results(dcase_output_val_dir)
    logger.info('ER/F/LE/LR/SELD: {}'.format('{:0.4f}/{:0.4f}/{:0.4f}/{:0.4f}/{:0.4f}'.format(val_ER, val_F, val_LE, val_LR, val_seld_scr)))
    print('ER/F/LE/LR/SELD: {}'.format('{:0.4f}/{:0.4f}/{:0.4f}/{:0.4f}/{:0.4f}'.format(val_ER, val_F, val_LE, val_LR, val_seld_scr)))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser('train')
    parser.add_argument('-c', '--config_name', type=str, default='foa_dev_multi_accdoa_nopool', help='name of config')
    input_args = parser.parse_args()

    with open(os.path.join('config', '{}.yaml'.format(input_args.config_name)), 'r') as f:
        args = yaml.safe_load(f)
    main(args)