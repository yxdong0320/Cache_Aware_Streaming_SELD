import os, shutil, argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import yaml
import pdb

from lmdb_data_loader_A import LmdbDataset

from models.resnet_conformer_audio import ResnetConformer_sed_doa_nopool
from utils.cls_tools.cls_compute_seld_results import ComputeSELDResults
from utils.write_csv import write_output_format_file

from utils.sed_doa import SedDoaResult, process_foa_input_sed_doa_labels, SedDoaLoss


def add_item(wav_name, seq_result, output_data):
    items = wav_name.split('_')
    csv_name = '_'.join(items[:-3])
    seg_cnt = int(items[-1])
    if csv_name not in output_data:
        output_data[csv_name] = {}
    output_data[csv_name][seg_cnt] = []
    output_data[csv_name][seg_cnt] = seq_result

    
def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return None

def main(args):
    # 设置log
    log_output_folder = os.path.dirname(args['result']['log_output_path'])
    os.makedirs(log_output_folder, exist_ok=True)
    logging.basicConfig(filename=args['result']['log_output_path'], filemode='w', level=logging.INFO, format='%(levelname)s: %(asctime)s: %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
    logger = logging.getLogger(__name__)
    logger.info(args)

    data_process_fn = process_foa_input_sed_doa_labels
    criterion = SedDoaLoss(loss_weight=[0.1,1])
    model = ResnetConformer_sed_doa_nopool(in_channel=args['model']['in_channel'], in_dim=args['model']['in_dim'], out_dim=args['model']['out_dim'])
    # 测试集初始化
    test_split = [4]
    test_dataset = LmdbDataset(args['data']['test_lmdb_dir'], test_split, normalized_features_wts_file=args['data']['norm_file'],
                                ignore=args['data']['test_ignore'], segment_len=args['data']['segment_len'], data_process_fn=data_process_fn)

    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=1, shuffle=False, 
        num_workers=args['train']['test_num_workers'], collate_fn=test_dataset.collater
    )

    # 模型初始化
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)
    logger.info(model)
    set_random_seed(12332)

    if args['model']['pre-train']:
        model_dict = model.state_dict()  # 获取当前模型的所有参数
        pretrained_dict = torch.load(args['model']['pre-train_model'], map_location=device)
        # key_list = [key for key in pretrained_dict.keys() if ('resnet_aug' not in key)] 
        key_list = [key for key in pretrained_dict.keys() if ('ManifoldMixup_resnet' not in key)] 
        # key_list = [key for key in pretrained_dict.keys()] # 更新模型参数，排除特定的层
        for key in key_list:
            model_dict[key] = pretrained_dict[key]
        model.load_state_dict(model_dict)

    epoch_count = 0
    step_count = 0
    test_loss = []
    
    # 测试
    start_time = time.time()
    model.eval()
    output_data = {}
    if args['model']['use_class_thre']:
        Classwise_thre = args['model']['class_thre']
    for data in tqdm(test_dataloader):
        input = data['input'].to(device)
        target = data['target'].to(device)
        wav_names = data['wav_names']
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)
            test_loss.append(loss.item())
            output = output.detach().cpu().numpy()
        for b, wav_name in enumerate(wav_names):
            add_item(wav_name, output[b], output_data)

    output_dict = {}
    segment_len = args['data']['segment_len']
    overlap_len = args['data']['overlap_len']
    hop_len = segment_len - overlap_len
    if args['model']['inf_type'] == 'past_future':
        past_len = args['model']['past_len']
        for i, filename in enumerate(output_data):
            length = len(output_data[filename])
            data = np.zeros(((length-1)*hop_len+segment_len, 52))
            data[:overlap_len,:] = output_data[filename][0][:overlap_len,:]
            for j in range(length-1):
                data[(j+1)*hop_len + past_len:(j+1)*hop_len+overlap_len,:] = output_data[filename][j+1][past_len:overlap_len,:]
            data[(length - 1)* hop_len + overlap_len:,:] = output_data[filename][length-1][overlap_len:,:]
            data[length*hop_len:,:] = output_data[filename][length-1][hop_len:,:]
            sed_pred = data[:, :13]
            doa_pred = data[:, 13:]
            if filename not in output_dict:
                output_dict[filename] = {}
            for frame_cnt in range(sed_pred.shape[0]):
                for class_cnt in range(sed_pred.shape[1]):
                    if args['model']['use_class_thre']:
                        if sed_pred[frame_cnt][class_cnt] > Classwise_thre[class_cnt]:
                            if frame_cnt not in output_dict[filename]:
                                output_dict[filename][frame_cnt] = []
                            output_dict[filename][frame_cnt].append([class_cnt, doa_pred[frame_cnt][class_cnt], doa_pred[frame_cnt][class_cnt+13], doa_pred[frame_cnt][class_cnt+26]])
                    else:
                        if sed_pred[frame_cnt][class_cnt] > 0.5:
                            if frame_cnt not in output_dict[filename]:
                                output_dict[filename][frame_cnt] = []
                            output_dict[filename][frame_cnt].append([class_cnt, doa_pred[frame_cnt][class_cnt], doa_pred[frame_cnt][class_cnt+13], doa_pred[frame_cnt][class_cnt+26]])
    else:
        for i, filename in enumerate(output_data):
            length = len(output_data[filename])
            data = np.zeros(((length-1)*hop_len+segment_len, 52))
            data[:hop_len,:] = output_data[filename][0][:hop_len,:]
            for j in range(length-1):
                data[(j+1)*hop_len:(j+1)*hop_len+overlap_len,:] = (output_data[filename][j][hop_len:,:] + output_data[filename][j+1][:overlap_len,:]) / 2
            
            data[length*hop_len:,:] = output_data[filename][length-1][hop_len:,:]
            sed_pred = data[:, :13]
            doa_pred = data[:, 13:]
            if filename not in output_dict:
                output_dict[filename] = {}
            for frame_cnt in range(sed_pred.shape[0]):
                for class_cnt in range(sed_pred.shape[1]):
                    if args['model']['use_class_thre']:
                        if sed_pred[frame_cnt][class_cnt] > Classwise_thre[class_cnt]:
                            if frame_cnt not in output_dict[filename]:
                                output_dict[filename][frame_cnt] = []
                            output_dict[filename][frame_cnt].append([class_cnt, doa_pred[frame_cnt][class_cnt], doa_pred[frame_cnt][class_cnt+13], doa_pred[frame_cnt][class_cnt+26]])
                    else:
                        if sed_pred[frame_cnt][class_cnt] > 0.5:
                            if frame_cnt not in output_dict[filename]:
                                output_dict[filename][frame_cnt] = []
                            output_dict[filename][frame_cnt].append([class_cnt, doa_pred[frame_cnt][class_cnt], doa_pred[frame_cnt][class_cnt+13], doa_pred[frame_cnt][class_cnt+26]])

    test_time = time.time() - start_time
    
    # 保存测试集CSV文件
    dcase_output_val_dir = os.path.join(args['result']['dcase_output_dir'], 'test')
    os.makedirs(dcase_output_val_dir, exist_ok=True)
    for csv_name, perfile_out_dict in output_dict.items():
        output_file = os.path.join(dcase_output_val_dir, '{}.csv'.format(csv_name))
        write_output_format_file(output_file, perfile_out_dict)

    score_obj = ComputeSELDResults(ref_files_folder=args['data']['ref_files_dir'])
    val_ER, val_F, val_LE, val_LR, val_seld_scr, classwise_val_scr = score_obj.get_SELD_Results(dcase_output_val_dir)
    logger.info('test_time:{:.2f}, average_test_loss:{:.4f}'.format(test_time,np.mean(test_loss)))
    logger.info('ER/F/LE/LR/SELD: {}'.format('{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}'.format(val_ER, val_F, val_LE, val_LR, val_seld_scr)))
    print('test_time:{:.2f}, average_test_loss:{:.4f}'.format(test_time,np.mean(test_loss)))
    print('ER/F/LE/LR/SELD: {}'.format('{:0.4f}/{:0.4f}/{:0.4f}/{:0.4f}/{:0.4f}'.format(val_ER, val_F, val_LE, val_LR, val_seld_scr)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser('train')
    parser.add_argument('-c', '--config_name', type=str, default='foa_dev_multi_accdoa_nopool', help='name of config')
    input_args = parser.parse_args()

    # 不同任务使用不同配置文件
    # foa_dev_seddoa_nopool
    # foa_dev_accdoa_nopool
    # foa_dev_multi_accdoa_nopool
    with open(os.path.join('config', '{}.yaml'.format(input_args.config_name)), 'r') as f:
        args = yaml.safe_load(f)
    main(args)
