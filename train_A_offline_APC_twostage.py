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

def main(args, apc_only=False):
    # 设置log
    log_output_folder = os.path.dirname(args['result']['log_output_path'])
    os.makedirs(log_output_folder, exist_ok=True)
    logging.basicConfig(filename=args['result']['log_output_path'], filemode='w', level=logging.INFO, format='%(levelname)s: %(asctime)s: %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
    logger = logging.getLogger(__name__)
    logger.info(args)
    
    if apc_only:
        logger.info("只使用APC损失进行预训练")
    else:
        logger.info("使用SELD和APC损失进行训练")

    data_process_fn = process_foa_input_sed_doa_labels
    result_class = SedDoaResult
    criterion = SedDoaLoss(loss_weight=[0.1,1])
    model = ResnetConformer_sed_doa_nopool(in_channel=args['model']['in_channel'], 
                                            in_dim=args['model']['in_dim'], 
                                            out_dim=args['model']['out_dim'],
                                            apc_future_steps=args['model']['apc_future_steps'],
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

    def load_partial_model(model, pretrained_path):
        """只加载resnet、input_projection和conformer_layers部分的权重"""
        pretrained_dict = torch.load(pretrained_path)
        model_dict = model.state_dict()
        
        # 过滤只保留特征提取层的权重
        filtered_dict = {}
        for k, v in pretrained_dict.items():
            # 只加载特征提取部分的权重
            if 'sed_out_layer' not in k and 'out_layer' not in k and 't_pooling' not in k:
                filtered_dict[k] = v
        
        # 打印加载的层数和总层数
        logger.info(f"从预训练模型加载 {len(filtered_dict)}/{len(pretrained_dict)} 层权重")
        
        # 更新当前模型的权重
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
        return model
    
    # # 方案2：在第二阶段只加载部分预训练权重
    # if args['model']['pre-train']:
    #     logger.info(f"加载预训练模型: {args['model']['pre-train_model']}")
    #     if apc_only:
    #         # 预训练阶段直接加载全部权重
    #         model.load_state_dict(torch.load(args['model']['pre-train_model']))
    #     else:
    #         # 微调阶段只加载特征提取部分权重
    #         model = load_partial_model(model, args['model']['pre-train_model'])
    #         logger.info("只加载了特征提取部分的预训练权重，输出层保持随机初始化")

    if args['model']['pre-train']:
        logger.info(f"加载预训练模型: {args['model']['pre-train_model']}")
        model.load_state_dict(torch.load(args['model']['pre-train_model']))
    # logger.info(model)
        
    # 差异化学习率设置
    # 方案4：在第二阶段的开始阶段使用更高的学习率
    if args['model']['pre-train']:
        logger.info(f"加载预训练模型: {args['model']['pre-train_model']}")
        model.load_state_dict(torch.load(args['model']['pre-train_model']))
        
        # 为不同部分设置不同的学习率
        feature_params = list(model.resnet.parameters()) + list(model.input_projection.parameters()) + list(model.conformer_layers.parameters())
        output_params = list(model.sed_out_layer.parameters()) + list(model.out_layer.parameters())
        
        optimizer = optim.Adam([
            {'params': feature_params, 'lr': args['train']['lr'] * 1.0},  # 特征提取部分低学习率
            {'params': output_params, 'lr': args['train']['lr'] * 20.0}    # 输出层高学习率
        ])
    else:
        optimizer = optim.Adam(model.parameters(), lr=args['train']['lr'])

    # 优化器初始化
    # optimizer = optim.Adam(model.parameters(), lr=args['train']['lr'])
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
    best_apc_loss = float('inf')  # 初始化最佳APC损失
    best_criterion = best_apc_loss if apc_only else best_seld_score  # 根据训练阶段选择评价指标
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
            
            # if apc_only:
            #     # 只使用APC损失进行预训练
            #     total_loss = apc_loss

            # 方案5 在第一阶段就小幅度使用SELD损失：
            if apc_only:
                # 在预训练阶段使用大权重的APC损失和小权重的SELD损失
                sed_doa_loss = criterion(output, target)
                total_loss = sed_doa_loss * 0.01 + apc_loss * 0.99  # 主要关注APC损失
                train_loss.append(sed_doa_loss.item())

            # # 方案3：在预训练阶段添加随机初始化的SELD输出
            # if apc_only:
            #     # 只使用APC损失进行预训练，但添加输出层随机梯度
            #     total_loss = apc_loss
                
            #     # 添加一个小的随机梯度给输出层，防止它们停留在初始化状态
            #     if step_count % 10 == 0:  # 每10步执行一次
            #         random_grad_scale = 0.01  # 设置一个很小的比例
            #         for param in list(model.sed_out_layer.parameters()) + list(model.out_layer.parameters()):
            #             if param.grad is None:
            #                 param.grad = torch.randn_like(param) * random_grad_scale
            #             else:
            #                 param.grad += torch.randn_like(param) * random_grad_scale
            else:
                # 同时使用SELD和APC损失
                sed_doa_loss = criterion(output, target)
                total_loss = sed_doa_loss + apc_loss * args['train']['apc_loss_weight']
                train_loss.append(sed_doa_loss.item())
                
            train_apc_loss.append(apc_loss.item())
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            
            step_count += 1
            if step_count % args['result']['log_interval'] == 0:
                lr = optimizer.param_groups[0]['lr']
                if apc_only:
                    logger.info('epoch: {}, step: {}/{}, lr:{:.6f}, apc_loss:{:.4f}'.format(
                        epoch_count, step_count, total_steps, lr, apc_loss.item()))
                else:
                    logger.info('epoch: {}, step: {}/{}, lr:{:.6f}, train_loss:{:.4f}, apc_loss:{:.4f}'.format(
                        epoch_count, step_count, total_steps, lr, sed_doa_loss.item(), apc_loss.item()))
            
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
                if not apc_only:
                    sed_doa_loss = criterion(output, target)
                    test_loss.append(sed_doa_loss.item())
                test_apc_loss.append(apc_loss.item())

            if not apc_only:
                test_result.add_items(data['wav_names'], output)
                
        test_time = time.time() - start_time
        
        if apc_only:
            # 预训练阶段：根据APC损失评估并保存模型
            current_apc_loss = np.mean(test_apc_loss)
            logger.info('epoch: {}, step: {}/{}, train_time:{:.2f}, test_time:{:.2f}, average_train_apc_loss:{:.4f}, average_test_apc_loss:{:.4f}'.format(
                epoch_count, step_count, total_steps, train_time, test_time, np.mean(train_apc_loss), current_apc_loss))
            
            # 保存模型
            checkpoint_output_dir = args['result']['checkpoint_output_dir']
            os.makedirs(checkpoint_output_dir, exist_ok=True)
            model_path = os.path.join(checkpoint_output_dir, 'checkpoint_epoch{}_step{}.pth'.format(epoch_count, step_count))
            torch.save(model.state_dict(), model_path)
            logger.info('save checkpoint: {}'.format(model_path))
            
            # 更新最佳性能记录
            if current_apc_loss < best_apc_loss:
                best_apc_loss = current_apc_loss
                best_epoch = epoch_count
                best_checkpoint = model_path
                patience_counter = 0  # 重置早停计数器
                # 复制一份作为最佳模型
                best_model_path = os.path.join(checkpoint_output_dir, 'best_checkpoint.pth')
                shutil.copyfile(model_path, best_model_path)
                logger.info('New best model found with APC loss: {:.4f}'.format(best_apc_loss))
            else:
                patience_counter += 1
                logger.info('No improvement for {} epochs. Best APC loss so far: {:.4f}'.format(
                    patience_counter, best_apc_loss))
        else:
            # 微调阶段：使用SELD分数评估
            output_dict = test_result.get_result()
            
            # 保存测试集CSV文件
            dcase_output_val_dir = os.path.join(args['result']['dcase_output_dir'], 'epoch{}_step{}'.format(epoch_count, step_count))
            os.makedirs(dcase_output_val_dir, exist_ok=True)
            for csv_name, perfile_out_dict in output_dict.items():
                output_file = os.path.join(dcase_output_val_dir, '{}.csv'.format(csv_name))
                write_output_format_file(output_file, perfile_out_dict)
            
            #根据保存的CSV文件进行结果评估
            score_obj = ComputeSELDResults(ref_files_folder=args['data']['ref_files_dir'])
            val_ER, val_F, val_LE, val_LR, val_seld_scr, classwise_val_scr = score_obj.get_SELD_Results(dcase_output_val_dir)
            logger.info('epoch: {}, step: {}/{}, train_time:{:.2f}, test_time:{:.2f}, average_train_loss:{:.4f}, average_test_loss:{:.4f}, average_train_apc_loss:{:.4f}, average_test_apc_loss:{:.4f}'.format(
                epoch_count, step_count, total_steps, train_time, test_time, np.mean(train_loss), np.mean(test_loss), np.mean(train_apc_loss), np.mean(test_apc_loss)))
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
                # 复制一份作为最佳模型
                best_model_path = os.path.join(checkpoint_output_dir, 'best_checkpoint.pth')
                shutil.copyfile(model_path, best_model_path)
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
    
    if apc_only:
        logger.info('APC loss: {:.4f}'.format(best_apc_loss))
    else:
        logger.info('SELD score: {:.4f}'.format(best_seld_score))
        
    logger.info('Checkpoint path: {}'.format(best_checkpoint))
    logger.info('Total epochs trained: {}'.format(epoch_count))
    logger.info('='*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('train')
    parser.add_argument('-c', '--config_name', type=str, default='foa_dev_multi_accdoa_nopool', help='name of config')
    parser.add_argument('--apc_only', action='store_true', help='only use APC loss for pre-training')
    input_args = parser.parse_args()
    # 不同任务使用不同配置文件
    with open(os.path.join('config', '{}.yaml'.format(input_args.config_name)), 'r') as f:
        args = yaml.safe_load(f)
    main(args, apc_only=input_args.apc_only)