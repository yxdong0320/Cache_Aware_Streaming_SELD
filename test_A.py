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

from models.cache_resnet_conformer import ResnetConformer_sed_doa_nopool
from utils.cls_tools.cls_compute_seld_results import ComputeSELDResults
from utils.write_csv import write_output_format_file

from utils.sed_doa import SedDoaResult, process_foa_input_sed_doa_labels, SedDoaLoss


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

    if args['model']['type'] == 'seddoa_nopool':
        data_process_fn = process_foa_input_sed_doa_labels
        result_class = SedDoaResult
        criterion = SedDoaLoss(loss_weight=[0.1,1])
        model = ResnetConformer_sed_doa_nopool(in_channel=args['model']['in_channel'], in_dim=args['model']['in_dim'], out_dim=args['model']['out_dim'])
    # 测试集初始化
    test_split = [4]
    test_dataset = LmdbDataset(args['data']['test_lmdb_dir'], test_split, normalized_features_wts_file=args['data']['norm_file'],
                                ignore=args['data']['test_ignore'], segment_len=args['data']['segment_len'], data_process_fn=data_process_fn)
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=32, shuffle=False, 
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
    test_result = result_class(segment_length=args['data']['segment_len'])
    for data in tqdm(test_dataloader):
        input = data['input'].to(device)
        target = data['target'].to(device)
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)
            test_loss.append(loss.item())

        test_result.add_items(data['wav_names'], output)
    output_dict = test_result.get_result()
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