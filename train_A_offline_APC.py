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

from lmdb_data_loader_A import LmdbDataset

from models.resnet_conformer_APC import ResnetConformer_sed_doa_nopool
from lr_scheduler.tri_stage_lr_scheduler import TriStageLRScheduler
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

    data_process_fn = process_foa_input_sed_doa_labels
    result_class = SedDoaResult
    criterion = SedDoaLoss(loss_weight=[0.1,1])
    apc_loss_f = args['model'].get('apc_loss_f', 'L1')
    print('apc_loss_f: {}'.format(apc_loss_f))

    model = ResnetConformer_sed_doa_nopool(in_channel=args['model']['in_channel'], 
                                                in_dim=args['model']['in_dim'], 
                                                out_dim=args['model']['out_dim'],
                                                apc_future_steps=args['model']['apc_future_steps'],
                                                loss_f=apc_loss_f,
                                                )

    # 训练集初始化
    train_split = [1,2,3]
    train_dataset = LmdbDataset(args['data']['train_lmdb_dir'], train_split, normalized_features_wts_file=args['data']['norm_file'],
                                ignore=args['data']['train_ignore'], segment_len=args['data']['segment_len'], data_process_fn=data_process_fn)
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=args['data']['batch_size'], shuffle=True, 
        num_workers=args['train']['train_num_workers'], collate_fn=train_dataset.collater
    )

    # 测试集初始化
    test_split = [4]
    test_dataset = LmdbDataset(args['data']['test_lmdb_dir'], test_split, normalized_features_wts_file=args['data']['norm_file'],
                                ignore=args['data']['test_ignore'], segment_len=args['data']['segment_len'], data_process_fn=data_process_fn)
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=args['data']['batch_size'], shuffle=False, 
        num_workers=args['train']['test_num_workers'], collate_fn=test_dataset.collater
    )

    # 模型初始化
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else "cpu")
    model = model.to(device)
    logger.info(model)
    set_random_seed(12332)

    if args['model']['pre-train']:
        model.load_state_dict(torch.load(args['model']['pre-train_model']))
    # logger.info(model)

    # 优化器初始化
    optimizer = optim.Adam(model.parameters(), lr=args['train']['lr'])
    total_steps = args['train']['nb_steps']
    warmup_steps = int(total_steps*0.1)
    hold_steps = int(total_steps*0.6)
    decay_steps = int(total_steps*0.3)
    scheduler = TriStageLRScheduler(optimizer, peak_lr=args['train']['lr'], init_lr_scale=0.01, final_lr_scale=0.05, 
                                    warmup_steps=warmup_steps, hold_steps=hold_steps, decay_steps=decay_steps)
    epoch_count = 0
    step_count = 0

    # 开始训练
    stop_training = False
    best_seld_score = float('inf')  # 初始化最佳SELD分数
    best_epoch = 0  # 初始化最佳epoch
    best_checkpoint = ''  # 初始化最佳checkpoint路径
    patience = args['train']['early_stop_patience']  # 早停耐心值
    patience_counter = 0  # 早停计数器
    while not stop_training:
        train_loss = []
        train_apc_loss = []
        test_loss = []
        test_apc_loss = []
        epoch_count += 1
        # 训练
        start_time = time.time()
        model.train()
        for data in train_dataloader:
            input = data['input'].to(device)
            target = data['target'].to(device)
            optimizer.zero_grad()
            output, apc_loss = model(input)
            loss = criterion(output, target)
            total_loss = loss + apc_loss * args['train']['apc_loss_weight']
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss.append(loss.item())
            train_apc_loss.append(apc_loss.item())
            step_count += 1
            if step_count % args['result']['log_interval'] == 0:
                lr = optimizer.param_groups[0]['lr']
                # logger.info('epoch: {}, step: {}/{}, lr:{:.6f}, train_loss:{:.4f}'.format(epoch_count, step_count, total_steps, lr, loss.item()))
                logger.info('epoch: {}, step: {}/{}, lr:{:.6f}, train_loss:{:.4f}, apc_loss:{:.4f}'.format(epoch_count, step_count, total_steps, lr, loss.item(), apc_loss.item()))
            if step_count >= total_steps:
                stop_training = True
                logger.info('Reached maximum number of steps')
                break
            
        torch.cuda.empty_cache()
        train_time = time.time() - start_time

        # 测试
        start_time = time.time()
        model.eval()
        test_result = result_class(segment_length=args['data']['segment_len'])
        for data in test_dataloader:
            input = data['input'].to(device)
            target = data['target'].to(device)
            with torch.no_grad():
                output, apc_loss = model(input)
                loss = criterion(output, target)
                test_loss.append(loss.item())
                test_apc_loss.append(apc_loss.item())

            test_result.add_items(data['wav_names'], output)
        output_dict = test_result.get_result()
        test_time = time.time() - start_time
        
        # 保存测试集CSV文件
        dcase_output_val_dir = os.path.join(args['result']['dcase_output_dir'], 'epoch{}_step{}'.format(epoch_count, step_count))
        os.makedirs(dcase_output_val_dir, exist_ok=True)
        for csv_name, perfile_out_dict in output_dict.items():
            output_file = os.path.join(dcase_output_val_dir, '{}.csv'.format(csv_name))
            write_output_format_file(output_file, perfile_out_dict)
        
        #根据保存的CSV文件进行结果评估
        score_obj = ComputeSELDResults(ref_files_folder=args['data']['ref_files_dir'])
        val_ER, val_F, val_LE, val_LR, val_seld_scr, classwise_val_scr = score_obj.get_SELD_Results(dcase_output_val_dir)
        logger.info('epoch: {}, step: {}/{}, train_time:{:.2f}, test_time:{:.2f}, average_train_loss:{:.4f}, average_test_loss:{:.4f}'.format(epoch_count, step_count, total_steps, train_time, test_time, np.mean(train_loss), np.mean(test_loss)))
        logger.info('ER/F/LE/LR/SELD: {}'.format('{:0.4f}/{:0.4f}/{:0.4f}/{:0.4f}/{:0.4f}'.format(val_ER, val_F, val_LE, val_LR, val_seld_scr)))
        # 保存模型
        checkpoint_output_dir = args['result']['checkpoint_output_dir']
        os.makedirs(checkpoint_output_dir, exist_ok=True)
        model_path = os.path.join(checkpoint_output_dir, 'checkpoint_epoch{}_step{}.pth'.format(epoch_count, step_count))
        torch.save(model.state_dict(), model_path)
        logger.info('save checkpoint: {}'.format(model_path))

        # 更新最佳性能记录
        if val_seld_scr < best_seld_score:
            best_seld_score = val_seld_scr
            best_epoch = epoch_count
            best_checkpoint = model_path
            patience_counter = 0  # 重置早停计数器
            logger.info('New best model found SELD score: {:.4f}'.format(best_seld_score))
        else:
            patience_counter += 1
            logger.info('No improvement for {} epochs. Best SELD score so far: {:.4f}'.format(
                patience_counter, best_seld_score))

                # 检查是否应该早停
        if patience_counter >= patience:
            logger.info('Early stopping triggered after {} epochs without improvement'.format(patience))
            stop_training = True

    # 训练结束后记录最佳性能
    logger.info('='*50)
    logger.info('Training completed!')
    if patience_counter >= patience:
        logger.info('Stopped due to: Early stopping criterion met')
    else:
        logger.info('Stopped due to: Maximum steps reached')
    logger.info('Best performance:')
    logger.info('Epoch: {}'.format(best_epoch))
    logger.info('SELD score: {:.4f}'.format(best_seld_score))
    logger.info('Checkpoint path: {}'.format(best_checkpoint))
    logger.info('Total epochs trained: {}'.format(epoch_count))
    logger.info('='*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('train')
    parser.add_argument('-c', '--config_name', type=str, default='foa_dev_multi_accdoa_nopool', help='name of config')
    input_args = parser.parse_args()
    # 不同任务使用不同配置文件
    with open(os.path.join('config', '{}.yaml'.format(input_args.config_name)), 'r') as f:
        args = yaml.safe_load(f)
    main(args)