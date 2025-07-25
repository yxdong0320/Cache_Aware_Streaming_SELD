from cgi import test
import os, shutil, argparse
import time
# from deeplearning.exp3.train import train
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import yaml
import pdb

from lmdb_data_loader_A import LmdbDataset

from models.dual_mode_cache_resnet_conformer_APC import DualModeResnetConformerAPC, DualModeResnetConformerCrossModalAPC
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
    model = DualModeResnetConformerCrossModalAPC(
        in_channel=args['model']['in_channel'], 
        in_dim=args['model']['in_dim'], 
        out_dim=args['model']['out_dim'],
        att_context_size=args['model']['att_context_size'],
        num_conformer_layer=args['model']['num_conformer_layer'],
        encoder_dim=args['model']['encoder_dim'],
        apc_future_steps=args['model']['apc_future_steps'],
        use_nonstream_apc=args['train'].get('use_nonstream_apc', False),  # 从train配置中读取
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

    def get_dynamic_weights(step_count, total_steps):
        """根据训练进度动态调整权重"""
        progress = step_count / total_steps
        
        if progress < 0.3:
            # 前期：平衡学习
            return {
                'stream_weight': 1.0,
                'distill_weight': 0.3,
                'apc_weight': 0.05
            }
        elif progress < 0.7:
            # 中期：增强流式
            return {
                'stream_weight': 1.5,
                'distill_weight': 0.2,
                'apc_weight': 0.1
            }
        else:
            # 后期：专注流式
            return {
                'stream_weight': 2.0,
                'distill_weight': 0.1,
                'apc_weight': 0.15
            }

    # 开始训练
    stop_training = False
    best_seld_score = float('inf')  # 初始化最佳SELD分数
    best_epoch = 0  # 初始化最佳epoch
    best_checkpoint = ''  # 初始化最佳checkpoint路径
    patience = args['train']['early_stop_patience']  # 早停耐心值
    patience_counter = 0  # 早停计数器
    while not stop_training:

        train_total_loss = []
        train_distill_loss = []
        train_stream_loss = []
        train_non_stream_loss = []
        
        # 根据APC配置初始化相应的loss记录
        if args['train'].get('use_APC', False):
            if args['train'].get('use_cross_modal_apc', False):
                train_cross_modal_apc_loss = []
            if args['train'].get('use_nonstream_apc', False):
                train_non_stream_apc_loss = []

        test_stream_loss = []
        test_non_stream_loss = []
        epoch_count += 1

        # 训练
        start_time = time.time()
        model.train()

        for data in train_dataloader:
            input = data['input'].to(device)
            target = data['target'].to(device)
            optimizer.zero_grad()
            outputs = model(input, mode='dual')
            stream_pred = outputs['stream_pred']
            nonstream_pred = outputs['nonstream_pred']
            distill_loss = outputs['distill_loss']
            
            stream_loss = criterion(stream_pred, target)
            nonstream_loss = criterion(nonstream_pred, target)

            if args['train'].get('tri_stage_loss_weight', False):
            # 在训练循环中使用
                weights = get_dynamic_weights(step_count, total_steps)
                total_loss = stream_loss * weights['stream_weight'] + \
                            nonstream_loss * args['train']['nonstream_weight'] + \
                            distill_loss * weights['distill_weight']

            else:
            # 基础损失
                total_loss = stream_loss * args['train']['stream_weight'] + \
                            nonstream_loss * args['train']['nonstream_weight'] + \
                            distill_loss * args['train']['distill_weight']
            
            # 根据配置添加APC损失
            if args['train'].get('use_APC', False):
                apc_losses = outputs['apc_losses']
                apc_loss_components = 0.0
                
                if args['train'].get('use_cross_modal_apc', False):
                    cross_modal_apc_loss = apc_losses['cross_modal_apc_loss']
                    apc_loss_components += cross_modal_apc_loss
                    train_cross_modal_apc_loss.append(cross_modal_apc_loss.item())
                
                if args['train'].get('use_nonstream_apc', False):
                    nonstream_apc_loss = apc_losses['nonstream_apc_loss']
                    apc_loss_components += nonstream_apc_loss
                    train_non_stream_apc_loss.append(nonstream_apc_loss.item())

                if args['train'].get('tri_stage_loss_weight', False):
                    total_loss += apc_loss_components * weights['apc_weight']
                else:
                    total_loss += apc_loss_components * args['train']['apc_weight']

            total_loss.backward()
            optimizer.step()
            scheduler.step()

            # 记录基础损失
            train_total_loss.append(total_loss.item())
            train_distill_loss.append(distill_loss.item())
            train_stream_loss.append(stream_loss.item())
            train_non_stream_loss.append(nonstream_loss.item())

            step_count += 1
            if step_count % args['result']['log_interval'] == 0:
                lr = optimizer.param_groups[0]['lr']
                
                # 动态构建日志信息
                log_msg = 'epoch: {}, step: {}/{}, lr:{:.6f}, train_loss:{:.4f}, train_distill_loss:{:.4f}, train_stream_loss:{:.4f}, train_non_stream_loss:{:.4f}'.format(
                    epoch_count, step_count, total_steps, lr, 
                    np.mean(train_total_loss), np.mean(train_distill_loss), 
                    np.mean(train_stream_loss), np.mean(train_non_stream_loss)
                )
                
                # 根据配置添加APC相关日志
                if args['train'].get('use_APC', False):
                    if args['train'].get('use_cross_modal_apc', False):
                        log_msg += ', train_cross_modal_apc_loss:{:.4f}'.format(np.mean(train_cross_modal_apc_loss))
                    if args['train'].get('use_nonstream_apc', False):
                        log_msg += ', train_non_stream_apc_loss:{:.4f}'.format(np.mean(train_non_stream_apc_loss))
                
                logger.info(log_msg)
                
            if step_count >= total_steps:
                stop_training = True
                logger.info('Reached maximum number of steps')
                break
            
        torch.cuda.empty_cache()
        train_time = time.time() - start_time

        # 测试
        start_time = time.time()
        model.eval()
        test_result_nonstream = result_class(segment_length=args['data']['segment_len'])
        test_result_stream = result_class(segment_length=args['data']['segment_len'])
        for data in test_dataloader:
            input = data['input'].to(device)
            target = data['target'].to(device)
            with torch.no_grad():
                nonstream_pred = model(input, mode='non-streaming')
                nonstream_loss_test = criterion(nonstream_pred, target)
                test_non_stream_loss.append(nonstream_loss_test.item())

                # 单独评估流式模式
                stream_pred = model(input, mode='streaming')  # 无缓存的流式评估
                stream_loss_test = criterion(stream_pred, target)
                test_stream_loss.append(stream_loss_test.item())

            test_result_nonstream.add_items(data['wav_names'], nonstream_pred)
            test_result_stream.add_items(data['wav_names'], stream_pred)
        output_dict_nonstream = test_result_nonstream.get_result()
        output_dict_stream = test_result_stream.get_result()
        test_time = time.time() - start_time
        # ... CSV文件保存和SELD评估代码保持不变 ...
        # 保存测试集CSV文件
        dcase_output_val_dir_nonstream = os.path.join(args['result']['dcase_output_dir'], 'epoch{}_step{}_nonstream'.format(epoch_count, step_count))
        os.makedirs(dcase_output_val_dir_nonstream, exist_ok=True)
        for csv_name, perfile_out_dict in output_dict_nonstream.items():
            output_file = os.path.join(dcase_output_val_dir_nonstream, '{}.csv'.format(csv_name))
            write_output_format_file(output_file, perfile_out_dict)

        dcase_output_val_dir_stream = os.path.join(args['result']['dcase_output_dir'], 'epoch{}_step{}_stream'.format(epoch_count, step_count))
        os.makedirs(dcase_output_val_dir_stream, exist_ok=True)
        for csv_name, perfile_out_dict in output_dict_stream.items():
            output_file = os.path.join(dcase_output_val_dir_stream, '{}.csv'.format(csv_name))
            write_output_format_file(output_file, perfile_out_dict)
        
        #根据保存的CSV文件进行结果评估
        score_obj = ComputeSELDResults(ref_files_folder=args['data']['ref_files_dir'])
        nonstream_ER, nonstream_F, nonstream_LE, nonstream_LR, nonstream_seld_scr, nonstream_classwise_scr = score_obj.get_SELD_Results(dcase_output_val_dir_nonstream)
        del score_obj
        score_obj = ComputeSELDResults(ref_files_folder=args['data']['ref_files_dir'])
        stream_ER, stream_F, stream_LE, stream_LR, stream_seld_scr, stream_classwise_scr = score_obj.get_SELD_Results(dcase_output_val_dir_stream)
        # 修改测试阶段的日志输出，只显示SELD task loss
        logger.info('epoch: {}, step: {}/{}, train_time:{:.2f}, test_time:{:.2f}, average_train_loss:{:.4f}, average_test_nonstream_loss:{:.4f}, average_test_stream_loss:{:.4f}'.format(
            epoch_count, step_count, total_steps, train_time, test_time, 
            np.mean(train_total_loss), np.mean(test_non_stream_loss), np.mean(test_stream_loss)
        ))
        logger.info('Nonstream ER/F/LE/LR/SELD: {}'.format('{:0.4f}/{:0.4f}/{:0.4f}/{:0.4f}/{:0.4f}'.format(nonstream_ER, nonstream_F, nonstream_LE, nonstream_LR, nonstream_seld_scr)))
        logger.info('Stream ER/F/LE/LR/SELD: {}'.format('{:0.4f}/{:0.4f}/{:0.4f}/{:0.4f}/{:0.4f}'.format(stream_ER, stream_F, stream_LE, stream_LR, stream_seld_scr)))
        
        # 保存模型
        checkpoint_output_dir = args['result']['checkpoint_output_dir']
        os.makedirs(checkpoint_output_dir, exist_ok=True)
        model_path = os.path.join(checkpoint_output_dir, 'checkpoint_epoch{}_step{}.pth'.format(epoch_count, step_count))
        torch.save(model.state_dict(), model_path)
        logger.info('save checkpoint: {}'.format(model_path))

        # 更新最佳性能记录
        if stream_seld_scr < best_seld_score:
            best_seld_score = stream_seld_scr
            best_epoch = epoch_count
            best_checkpoint = model_path
            patience_counter = 0  # 重置早停计数器
            logger.info('New best stream model found SELD score: {:.4f}'.format(best_seld_score))
        else:
            patience_counter += 1
            logger.info('No improvement for {} epochs. Best stream SELD score so far: {:.4f}'.format(
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