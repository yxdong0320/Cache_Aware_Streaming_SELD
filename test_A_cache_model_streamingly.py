import os, shutil, argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import yaml
import pdb
import joblib
import cls_feature_class
import parameters
import librosa
from tqdm import tqdm

from models.cache_resnet_conformer import ResnetConformer_sed_doa_nopool
from lr_scheduler.tri_stage_lr_scheduler import TriStageLRScheduler
from utils.cls_tools.cls_compute_seld_results import ComputeSELDResults
from utils.write_csv import write_output_format_file
from utils.sed_doa import SedDoaResult, process_foa_input_sed_doa, SedDoaLoss, SedDoaResult_Streaming_Inf

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return None

def read_wav_in_sliding_windows(wav_file, segment_length_sec=10, hop_length_sec=10):
    """
    按指定块大小顺序读取音频，最后一段不足长度时补零。
    
    Args:
        wav_file: 音频文件路径
        segment_length_sec: 每段音频的长度（秒）
        hop_length_sec: 移动步长（秒），建议与segment_length_sec保持相同
    
    Returns:
        generator，每次返回：
        - audio_segment: 音频数据，shape为[num_channels, segment_length_samples]
        - start_time: 当前段的开始时间（秒）
        - end_time: 当前段的结束时间（秒）
        - is_last_segment: 是否为最后一段
    """
    # 使用librosa加载音频，保持多通道
    audio, sr = librosa.load(wav_file, sr=None, mono=False)
    
    # 计算每个段的样本数
    segment_length_samples = int(segment_length_sec * sr)
    hop_length_samples = int(hop_length_sec * sr)
    
    # 获取音频的总长度和通道数
    num_channels = audio.shape[0]
    audio_length_samples = audio.shape[1]
    
    # 从0开始，以hop_length_samples为步长
    for start_sample in range(0, audio_length_samples, hop_length_samples):
        end_sample = start_sample + segment_length_samples
        is_last_segment = (start_sample + hop_length_samples >= audio_length_samples)
        
        if end_sample > audio_length_samples:
            # 创建一个全零数组，大小为[num_channels, segment_length_samples]
            padded_segment = np.zeros((num_channels, segment_length_samples))
            # 将剩余的音频数据复制到开头
            padded_segment[:, :audio_length_samples-start_sample] = audio[:, start_sample:]
            audio_segment = padded_segment
        else:
            audio_segment = audio[:, start_sample:end_sample]
        
        # 计算实际的时间范围
        start_time = start_sample / sr
        end_time = min(end_sample, audio_length_samples) / sr
        
        yield audio_segment, start_time, end_time, is_last_segment
        
        if is_last_segment:
            break

def main(args):
    # 设置log
    log_output_folder = os.path.dirname(args['result']['log_output_path'])
    os.makedirs(log_output_folder, exist_ok=True)
    logging.basicConfig(filename=args['result']['log_output_path'], filemode='w', level=logging.INFO, format='%(levelname)s: %(asctime)s: %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
    logger = logging.getLogger(__name__)
    logger.info(args)

    data_process_fn = process_foa_input_sed_doa
    result_class = SedDoaResult_Streaming_Inf
    criterion = SedDoaLoss(loss_weight=[0.1,1])

    normalized_features_wts_file = args['data']['norm_file']
    spec_scaler = joblib.load(normalized_features_wts_file)
    wav_path = args['data']['wav_folder_path']
    segment_length = args['model']['att_context_size'][1] + 1  
    segment_length_sec = segment_length * 0.02
    batch_size = args['data']['batch_size']

    params = parameters.get_params()
    dev_feat_cls = cls_feature_class.FeatureClass(params)

    model = ResnetConformer_sed_doa_nopool(in_channel=args['model']['in_channel'], 
                                                in_dim=args['model']['in_dim'], 
                                                out_dim=args['model']['out_dim'],
                                                att_context_size = args['model']['att_context_size'],
                                                num_conformer_layer = args['model']['num_conformer_layer'],
                                                encoder_dim=args['model']['encoder_dim'])

    # 模型初始化
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else "cpu")
    model = model.to(device)
    logger.info(model)
    set_random_seed(12332)

    if args['model']['pre-train']:
        model.load_state_dict(torch.load(args['model']['pre-train_model']))
    # logger.info(model)

    model.eval()
    test_result = result_class()

    for file in tqdm(os.listdir(wav_path)):
        # 清除GPU缓存
        if use_cuda:
            torch.cuda.empty_cache()
        file_path = os.path.join(wav_path, file)
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        # print(file_name)

        resnet_cache = model.get_initial_cache_resnet(batch_size)
        conformer_cache = model.get_initial_cache_conformer(batch_size)

        streaming_outputs = []

        for audio_segment, start_time, end_time, is_last_segment in read_wav_in_sliding_windows(
            file_path, segment_length_sec=segment_length_sec, hop_length_sec=segment_length_sec):

            data = dev_feat_cls.extract_audio_feature(audio_segment)
            # 特征规整
            data = spec_scaler.transform(data)
            data = data_process_fn(data)
            data = data.astype(np.float32)
            data = torch.from_numpy(data).to(device) # (7, segment_length, 64)
            data = data.unsqueeze(0) # (1, 7, segment_length, 64)

            with torch.no_grad():
                output, (resnet_cache, conformer_cache) = model(
                data, 
                resnet_cache=resnet_cache, 
                conformer_cache=conformer_cache)
            streaming_outputs.append(output)
        streaming_output = torch.cat(streaming_outputs, dim=1)
        test_result.add_items(file_name, streaming_output)

        del streaming_outputs, streaming_output

    output_dict = test_result.get_result()

    dcase_output_val_dir = args['result']['dcase_output_dir']
    os.makedirs(dcase_output_val_dir, exist_ok=True)
    for csv_name, perfile_out_dict in output_dict.items():
        output_file = os.path.join(dcase_output_val_dir, '{}.csv'.format(csv_name))
        write_output_format_file(output_file, perfile_out_dict)

    #根据保存的CSV文件进行结果评估
    score_obj = ComputeSELDResults(ref_files_folder=args['data']['ref_files_dir'])
    val_ER, val_F, val_LE, val_LR, val_seld_scr, classwise_val_scr = score_obj.get_SELD_Results(dcase_output_val_dir)
    logger.info('ER/F/LE/LR/SELD: {}'.format('{:0.4f}/{:0.4f}/{:0.4f}/{:0.4f}/{:0.4f}'.format(val_ER, val_F, val_LE, val_LR, val_seld_scr)))
    print('ER/F/LE/LR/SELD: {}'.format('{:0.4f}/{:0.4f}/{:0.4f}/{:0.4f}/{:0.4f}'.format(val_ER, val_F, val_LE, val_LR, val_seld_scr)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser('train')
    parser.add_argument('-c', '--config_name', type=str, default='foa_dev_multi_accdoa_nopool', help='name of config')
    input_args = parser.parse_args()
    # 不同任务使用不同配置文件
    with open(os.path.join('config', '{}.yaml'.format(input_args.config_name)), 'r') as f:
        args = yaml.safe_load(f)
    main(args) 








