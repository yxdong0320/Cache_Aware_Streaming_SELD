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
import time
import statistics
from collections import defaultdict

from models.cache_resnet_conformer import ResnetConformer_sed_doa_nopool
from utils.sed_doa import SedDoaResult, process_foa_input_sed_doa, SedDoaLoss, SedDoaResult_Streaming_Inf

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return None

def output_numpy_to_dict(input, class_thre = 0.5 * np.ones(13)):
    output_dict = {}
    sed = input[:,:,:13]
    doa = input[:,:,13:52]
    sed_pred = sed.squeeze(0)
    doa_pred = doa.squeeze(0)
    for frame_cnt in range(sed_pred.shape[0]):
        for class_cnt in range(sed_pred.shape[1]):
            if sed_pred[frame_cnt][class_cnt] > class_thre[class_cnt]:
                if frame_cnt not in output_dict:
                    output_dict[frame_cnt] = []
                output_dict[frame_cnt].append([class_cnt, doa_pred[frame_cnt][class_cnt], doa_pred[frame_cnt][class_cnt+13], doa_pred[frame_cnt][class_cnt+26]])
    return output_dict

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

class TimingStats:
    def __init__(self):
        self.feature_times = []
        self.inference_times = []
        self.postprocess_times = []
        
    def add_timing(self, stage, time_taken):
        if stage == 'feature':
            self.feature_times.append(time_taken)
        elif stage == 'inference':
            self.inference_times.append(time_taken)
        elif stage == 'postprocess':
            self.postprocess_times.append(time_taken)
    
    def get_statistics(self):
        def calc_stats(times):
            if not times:
                return None
            times = times[1:]  # 删除第一次运行的结果
            mean = statistics.mean(times)
            std = statistics.stdev(times) if len(times) > 1 else 0
            return {
                'mean': mean,
                'std': std,
                'min': min(times),
                'max': max(times),
                'median': statistics.median(times)
            }
        
        return {
            'feature': calc_stats(self.feature_times),
            'inference': calc_stats(self.inference_times),
            'postprocess': calc_stats(self.postprocess_times)
        }

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
    time_seq_output = 0.1

    params = parameters.get_params()
    dev_feat_cls = cls_feature_class.FeatureClass(params)

    model = ResnetConformer_sed_doa_nopool(in_channel=args['model']['in_channel'], 
                                                in_dim=args['model']['in_dim'], 
                                                out_dim=args['model']['out_dim'],
                                                att_context_size = args['model']['att_context_size'],
                                                num_conformer_layer = args['model']['num_conformer_layer'],
                                                encoder_dim=args['model']['encoder_dim'],)

    def warmup_gpu():
        """GPU预热"""
        dummy_input = torch.randn(1, 7, 500, 64).cuda()
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        torch.cuda.synchronize()

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

    # 初始化计时统计
    timing_stats = TimingStats()
    
    # GPU预热
    if use_cuda:
        warmup_gpu()

    results_dict = {}

    for file in tqdm(os.listdir(wav_path)):
        # 清除GPU缓存
        if use_cuda:
            torch.cuda.empty_cache()
        file_path = os.path.join(wav_path, file)
        file_name = os.path.splitext(os.path.basename(file_path))[0]

        results_dict[file_name] = []

        resnet_cache = model.get_initial_cache_resnet(batch_size)
        conformer_cache = model.get_initial_cache_conformer(batch_size)

        for audio_segment, start_time, end_time, is_last_segment in read_wav_in_sliding_windows(
            file_path, segment_length_sec=segment_length_sec, hop_length_sec=segment_length_sec):

            start_time_feature = time.perf_counter()
            data = dev_feat_cls.extract_audio_feature(audio_segment)
            # 特征规整
            data = spec_scaler.transform(data)
            data = data_process_fn(data)
            data = data.astype(np.float32)
            data = torch.from_numpy(data).to(device) # (7, segment_length, 64)
            data = data.unsqueeze(0) # (1, 7, segment_length, 64)
            torch.cuda.synchronize()
            feature_time = time.perf_counter() - start_time_feature
            timing_stats.add_timing('feature', feature_time)

            start_time_inference = time.perf_counter()
            with torch.no_grad():
                output, (resnet_cache, conformer_cache) = model(
                data, 
                resnet_cache=resnet_cache, 
                conformer_cache=conformer_cache)
            torch.cuda.synchronize()
            inference_time = time.perf_counter() - start_time_inference
            timing_stats.add_timing('inference', inference_time)

            start_time_postprocess = time.perf_counter()
            output_dict = output_numpy_to_dict(output)
            for _frame_ind in output_dict.keys():
                for _value in output_dict[_frame_ind]:
                    occur_time = (_frame_ind + 1) * time_seq_output + start_time
                    occur_class = int(_value[0])
                    x, y, z = float(_value[1]), float(_value[2]), float(_value[3])
                    result = {
                        "type": "inference_result",
                        "filename": file_name,
                        "timestamp": occur_time,
                        "result": f"Detected Sound Event Class {occur_class}: x {x:.2f},y {y:.2f},z {z:.2f}"
                    }
                    results_dict[file_name].append(result)
            postprocess_time = time.perf_counter() - start_time_postprocess
            timing_stats.add_timing('postprocess', postprocess_time)

    # 输出统计结果
    stats = timing_stats.get_statistics()
    logger.info("Timing Statistics:")
    logger.info(f"Feature Extraction: {stats['feature']}")
    logger.info(f"Model Inference: {stats['inference']}")
    logger.info(f"Post-processing: {stats['postprocess']}")
    
    total_latency = (stats['feature']['mean'] + 
                    stats['inference']['mean'] + 
                    stats['postprocess']['mean'] + 
                    segment_length_sec)
    logger.info(f"Total Latency (including lookahead): {total_latency:.4f} seconds")

    return results_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser('train')
    parser.add_argument('-c', '--config_name', type=str, default='foa_dev_multi_accdoa_nopool', help='name of config')
    input_args = parser.parse_args()
    # 不同任务使用不同配置文件
    with open(os.path.join('config', '{}.yaml'.format(input_args.config_name)), 'r') as f:
        args = yaml.safe_load(f)
    results_dict = main(args)
    
