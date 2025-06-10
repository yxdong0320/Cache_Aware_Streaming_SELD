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

def create_state_dict_mapping(pretrained_dict, model_dict):
    new_state_dict = {}
    
    # 创建conformer层的键值映射关系
    conformer_mapping = {
        'net.0': 'conv1',
        'net.2': 'conv2',
        'net.4.conv': 'depth_conv.conv',
        'net.7': 'norm'
    }
    
    for old_key in pretrained_dict.keys():
        # 跳过ManifoldMixup相关的层
        if 'ManifoldMixup_resnet' in old_key:
            continue
            
        # 处理conformer层
        if 'conformer_layers' in old_key:
            for old_pattern, new_pattern in conformer_mapping.items():
                if old_pattern in old_key:
                    new_key = old_key.replace(f'net.{old_pattern}', new_pattern)
                    if new_key in model_dict:
                        new_state_dict[new_key] = pretrained_dict[old_key]
        # 保留其他层的原始键值
        elif old_key in model_dict:
            new_state_dict[old_key] = pretrained_dict[old_key]
    
    return new_state_dict

def compare_model_keys(current_model, pretrained_dict):
    current_keys = set(current_model.state_dict().keys())
    pretrained_keys = set(pretrained_dict.keys())
    
    # 找出预训练模型中多出的键
    unexpected_keys = pretrained_keys - current_keys
    # 找出当前模型中缺失的键
    missing_keys = current_keys - pretrained_keys
    
    print("Unexpected keys in pretrained model:")
    for key in unexpected_keys:
        print(f"  {key}")
    
    print("\nMissing keys in current model:")
    for key in missing_keys:
        print(f"  {key}")
    
    return unexpected_keys, missing_keys

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return None

# 验证加载是否成功
def verify_loading(model, mapped_dict):
    current_state = model.state_dict()
    loaded_keys = set(mapped_dict.keys())
    model_keys = set(current_state.keys())
    
    print(f"Parameters loaded: {len(loaded_keys)}")
    print(f"Parameters in model: {len(model_keys)}")
    print(f"Parameters not loaded: {len(model_keys - loaded_keys)}")
    
    # 验证加载的参数值是否正确
    for key in loaded_keys:
        if not torch.equal(current_state[key], mapped_dict[key]):
            print(f"Warning: Parameter mismatch for {key}")

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
        model = ResnetConformer_sed_doa_nopool(in_channel=args['model']['in_channel'], 
                                                    in_dim=args['model']['in_dim'], 
                                                    out_dim=args['model']['out_dim'],
                                                    att_context_size = [100,49])
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
        model_dict = model.state_dict()
        pretrained_dict = torch.load(args['model']['pre-train_model'], map_location=device)
        
        # 创建映射后的状态字典
        mapped_dict = create_state_dict_mapping(pretrained_dict, model_dict)
        verify_loading(model, mapped_dict)
        
        # 更新模型参数
        model_dict.update(mapped_dict)
        
        # 打印加载统计信息
        print(f"Successfully loaded {len(mapped_dict)} parameters")
        print(f"Total parameters in current model: {len(model_dict)}")
        
        # 加载参数
        model.load_state_dict(model_dict, strict=False)

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