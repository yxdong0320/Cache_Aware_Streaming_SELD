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

from models.resnet_conformer_audio import ResnetConformer_sed_doa_nopool
from lr_scheduler.tri_stage_lr_scheduler import TriStageLRScheduler
from utils.cls_tools.cls_compute_seld_results import ComputeSELDResults
from utils.write_csv import write_output_format_file
from utils.sed_doa import SedDoaResult, process_foa_input_sed_doa, SedDoaLoss, SedDoaResult_Streaming_Inf

def read_wav_in_sliding_windows(wav_file, segment_length_sec=10, hop_length_sec=5):
    # 使用librosa加载音频，保持多通道音频
    audio, sr = librosa.load(wav_file, sr=None, mono=False)  # mono=False 保持原始通道数
    
    # 计算每个段的样本数
    segment_length_samples = segment_length_sec * sr
    hop_length_samples = int(hop_length_sec * sr)
    
    # 获取音频的总长度
    audio_length_samples = audio.shape[-1]  # audio.shape[1] 是样本数（多通道时，维度为 [num_channels, num_samples]）
    # pdb.set_trace()
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
    logging.basicConfig(filename=args['result']['log_output_path'], 
                        filemode='w', level=logging.INFO, 
                        format='%(levelname)s: %(asctime)s: %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
    logger = logging.getLogger(__name__)
    logger.info(args)

    model = ResnetConformer_sed_doa_nopool(in_channel=args['model']['in_channel'], 
                                           in_dim=args['model']['in_dim'], out_dim=args['model']['out_dim'])
    # result_class = SedDoaResult_Streaming_Inf
    data_process_fn = process_foa_input_sed_doa
    normalized_features_wts_file = args['data']['norm_file']
    spec_scaler = joblib.load(normalized_features_wts_file)
    wav_path = args['data']['wav_path']

    
    window_length = args['model']['window_length']  # 2
    segment_length = args['model']['segment_length']  # 10
    output_num_frame = args['model']['output_num_frame'] # 10
    time_seq_output = 1.0 / output_num_frame
    past_time = int(segment_length - window_length)
    window_len = int(window_length) * output_num_frame # 20
    past_len = int((segment_length - window_length) * output_num_frame) # 80
    sr = args['model']['sr']

    params = parameters.get_params()
    dev_feat_cls = cls_feature_class.FeatureClass(params)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    def warmup_gpu():
        """GPU预热"""
        dummy_input = torch.randn(1, 7, 500, 64).cuda()
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        torch.cuda.synchronize()
    model = model.to(device)
    logger.info(model)

    if args['model']['pre-train']:
        model.load_state_dict(torch.load(args['model']['pre-train_model']))

    model.eval()

    timing_stats = TimingStats()
    
    # GPU预热
    if use_cuda:
        warmup_gpu()
    # test_result = result_class()
    results_dict = {}

    for file in tqdm(os.listdir(wav_path)):
        if use_cuda:
            torch.cuda.empty_cache()
        file_path = os.path.join(wav_path, file)
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        results_dict[file_name] = []
        # 存储每一段音频的前半段输出
        for audio_segment, start_time, end_time, audio_len_samp in read_wav_in_sliding_windows(file_path, 
                                                                            segment_length_sec=segment_length, 
                                                                            hop_length_sec=window_length):
            output_now = []
            output_dict = {}
            # 获取当前段的真实长度
            current_segment_length = audio_segment.shape[-1] / sr  # 当前段的长度（单位：秒）
            
            if current_segment_length < 0.1:
                continue  # 跳过当前音频段                
            # 提取音频特征
            start_time_feature = time.perf_counter()
            data = dev_feat_cls.extract_audio_feature(audio_segment)
            # 特征规整
            data = spec_scaler.transform(data)
            data = data_process_fn(data)
            data = data.astype(np.float32)
            data = torch.from_numpy(data).to(device) # (7, 500, 64)
            data = data.unsqueeze(0) # (1, 7, 500, 64)
            torch.cuda.synchronize()
            feature_time = time.perf_counter() - start_time_feature
            timing_stats.add_timing('feature', feature_time)
            # pdb.set_trace()
            start_time_inference = time.perf_counter()
            with torch.no_grad():
                # 得到当前段的模型输出
                output = model(data) # torch.Size([1, 100, 52])
            torch.cuda.synchronize()
            inference_time = time.perf_counter() - start_time_inference
            timing_stats.add_timing('inference', inference_time)

            start_time_postprocess = time.perf_counter()
            if start_time == 0:
                # 首段：保留前overlap_len帧
                output_now = output
                time_offset = start_time
            elif end_time >= audio_len_samp/sr:  # 最后一段
                output_now = output[:, past_len:, :]
                time_offset = start_time + past_time         
            else:
                output_now = output[:, past_len:, :]
                time_offset = start_time + past_time 
            output_dict = output_numpy_to_dict(output_now)
            for _frame_ind in output_dict.keys():
                for _value in output_dict[_frame_ind]:
                    occur_time = (_frame_ind + 1) * time_seq_output + time_offset
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

    stats = timing_stats.get_statistics()
    logger.info("Timing Statistics:")
    logger.info(f"Feature Extraction: {stats['feature']}")
    logger.info(f"Model Inference: {stats['inference']}")
    logger.info(f"Post-processing: {stats['postprocess']}")
    
    total_latency = (stats['feature']['mean'] + 
                    stats['inference']['mean'] + 
                    stats['postprocess']['mean'] + 
                    window_length)
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