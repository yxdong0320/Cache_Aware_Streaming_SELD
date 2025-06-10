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
import torch.nn.functional as F

from lmdb_data_loader_A import LmdbDataset

from models.cache_resnet_conformer_TS import ResnetConformer_sed_doa_nopool_TS_hidstate_att_loss
from lr_scheduler.tri_stage_lr_scheduler import TriStageLRScheduler
from utils.cls_tools.cls_compute_seld_results import ComputeSELDResults
from utils.write_csv import write_output_format_file
from utils.sed_doa import SedDoaResult, process_foa_input_sed_doa_labels, SedDoaLoss, SedDoaKLLoss_2
from utils.sed_doa import HiddenStateMSELoss, AttentionMapMSELoss
from utils.sed_doa import HiddenStateMSELoss_norm, SimpleAttentionDivergenceLoss


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
    # 根据配置文件控制是否使用隐藏层和注意力图蒸馏
    use_hidden_distill = args['train'].get('use_hidden_distill', True)
    use_attn_distill = args['train'].get('use_attn_distill', True)
    use_ts_distill = args['train'].get('use_ts_distill', True)

    criterion = SedDoaLoss(loss_weight=[0.1,1])
    # 添加隐藏层和注意力图的损失函数
    if use_hidden_distill:
        hidden_criterion = HiddenStateMSELoss(loss_weight=args['train'].get('hidden_loss_weight', 0.2))
        # hidden_criterion = HiddenStateMSELoss_norm(loss_weight=args['train'].get('hidden_loss_weight', 0.2))
    if use_attn_distill:
        attn_criterion = AttentionMapMSELoss(loss_weight=args['train'].get('attn_loss_weight', 0.05))
        # attn_criterion = SimpleAttentionDivergenceLoss(loss_weight=args['train'].get('attn_loss_weight', 0.05))
    if use_ts_distill:
        kl_criterion = SedDoaKLLoss_2(loss_weight=[0.1, 1]) 
   
    model = ResnetConformer_sed_doa_nopool_TS_hidstate_att_loss(
            in_channel=args['model']['in_channel'], 
            in_dim=args['model']['in_dim'], 
            out_dim=args['model']['out_dim'],
            att_context_size=args['model']['att_context_size'],
            num_conformer_layer=args['model']['num_conformer_layer'],
            encoder_dim=args['model']['encoder_dim'],
            use_hidden_distill=use_hidden_distill,
            use_attn_distill=use_attn_distill,
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

    # 模型初始化后，加载预训练权重前的测试
    if args['model']['pre-train']:
        model_dict = model.state_dict()  # 获取当前模型的所有参数
        model_dict_t = model.teacher_model.state_dict()
        pretrained_dict_t = torch.load(args['model']['pre-train_model_t'], map_location=device)
        pretrained_dict_s = torch.load(args['model']['pre-train_model_s'], map_location=device)
        key_list = [key for key in pretrained_dict_s.keys()]  
        for key in key_list:
            model_dict[key] = pretrained_dict_s[key]
        key_list_t = [key for key in pretrained_dict_t.keys()] 
        for key in key_list_t:
            model_dict_t[key] = pretrained_dict_t[key]
        model.load_state_dict(model_dict)
        model.teacher_model.load_state_dict(model_dict_t)

        print('成功加载教师和学生模型')
        # logger.info("====== Testing teacher and student models before training ======")
        # # 冻结所有参数进行评估
        # for param in model.parameters():
        #     param.requires_grad = False
            
        # # 教师模型评估
        # model.eval()
        # test_result_teacher = result_class(segment_length=args['data']['segment_len'])
        # with torch.no_grad():
        #     for data in test_dataloader:
        #         input = data['input'].to(device)
        #         target = data['target'].to(device)
                
        #         # 仅使用教师模型
        #         teacher_output = model.teacher_model(input)
        #         teacher_loss = criterion(teacher_output, target)
                
        #         logger.info(f"Teacher model - Sample loss: {teacher_loss.item():.6f}")
        #         test_result_teacher.add_items(data['wav_names'], teacher_output)
                
        # # 评估教师模型性能
        # output_dict_teacher = test_result_teacher.get_result()
        # dcase_output_val_dir_teacher = os.path.join(args['result']['dcase_output_dir'], 'teacher_initial_test')
        # os.makedirs(dcase_output_val_dir_teacher, exist_ok=True)
        # for csv_name, perfile_out_dict in output_dict_teacher.items():
        #     output_file = os.path.join(dcase_output_val_dir_teacher, '{}.csv'.format(csv_name))
        #     write_output_format_file(output_file, perfile_out_dict)
        
        # score_obj = ComputeSELDResults(ref_files_folder=args['data']['ref_files_dir'])
        # t_ER, t_F, t_LE, t_LR, t_seld_scr, _ = score_obj.get_SELD_Results(dcase_output_val_dir_teacher)
        # logger.info(f"Teacher model initial performance - ER/F/LE/LR/SELD: {t_ER:.4f}/{t_F:.4f}/{t_LE:.4f}/{t_LR:.4f}/{t_seld_scr:.4f}")
        
        # # 学生模型评估（使用预训练权重加载前的状态）
        # test_result_student = result_class(segment_length=args['data']['segment_len'])
        # with torch.no_grad():
        #     for data in test_dataloader:
        #         input = data['input'].to(device)
        #         target = data['target'].to(device)
                
        #         # 获取学生模型输出（不包括知识蒸馏部分）
        #         if use_hidden_distill and use_attn_distill:
        #             output, _, _, _ = model(input)
        #         elif use_hidden_distill or use_attn_distill:
        #             output, _, _ = model(input)
        #         else:
        #             output, _ = model(input)
                    
        #         student_loss = criterion(output, target)
        #         logger.info(f"Student model - Sample loss: {student_loss.item():.6f}")
        #         test_result_student.add_items(data['wav_names'], output)
                
        # # 评估学生模型性能
        # output_dict_student = test_result_student.get_result()
        # dcase_output_val_dir_student = os.path.join(args['result']['dcase_output_dir'], 'student_initial_test')
        # os.makedirs(dcase_output_val_dir_student, exist_ok=True)
        # for csv_name, perfile_out_dict in output_dict_student.items():
        #     output_file = os.path.join(dcase_output_val_dir_student, '{}.csv'.format(csv_name))
        #     write_output_format_file(output_file, perfile_out_dict)
        
        # s_ER, s_F, s_LE, s_LR, s_seld_scr, _ = score_obj.get_SELD_Results(dcase_output_val_dir_student)
        # logger.info(f"Student model initial performance - ER/F/LE/LR/SELD: {s_ER:.4f}/{s_F:.4f}/{s_LE:.4f}/{s_LR:.4f}/{s_seld_scr:.4f}")
        
        # # 冻结教师模型参数
        # for param in model.teacher_model.parameters():
        #     param.requires_grad = False
            
        # # 确保学生模型参数可训练
        # for name, param in model.named_parameters():
        #     if not name.startswith('teacher_model.'):
        #         param.requires_grad = True

    # 优化器初始化
    optimizer = optim.Adam(model.parameters(), lr=args['train']['lr'])
    total_steps = args['train']['nb_steps']
    warmup_steps = int(total_steps*0.05)
    hold_steps = int(total_steps*0.25)
    decay_steps = int(total_steps*0.70)
    scheduler = TriStageLRScheduler(optimizer, peak_lr=args['train']['lr'], init_lr_scale=0.01, final_lr_scale=0.05, 
                                    warmup_steps=warmup_steps, hold_steps=hold_steps, decay_steps=decay_steps)
    epoch_count = 0
    step_count = 0

    # 开始训练
    stop_training = False
    best_seld_score = float('inf')  # 初始化最佳SELD分数
    best_epoch = 0  # 初始化最佳epoch
    best_checkpoint = ''  # 初始化最佳checkpoint路径
    patience = 80  # 早停耐心值
    patience_counter = 0  # 早停计数器
    while not stop_training:
        train_loss = []
        test_loss = []
        epoch_count += 1
        # 训练
        start_time = time.time()
        model.train()
        for data in train_dataloader:
            input = data['input'].to(device)
            target = data['target'].to(device)
            optimizer.zero_grad()

            # 根据模型配置获取不同的输出
            if use_hidden_distill and use_attn_distill:
                output, target_ts, (teacher_hidden, student_hidden), (teacher_attns, student_attns) = model(input)
            elif use_hidden_distill:
                output, target_ts, (teacher_hidden, student_hidden) = model(input)
            elif use_attn_distill:
                output, target_ts, (teacher_attns, student_attns) = model(input)
            else:
                output, target_ts = model(input)

            # # 计算各项损失
            # task_loss = criterion(output, target)
            # print(f"Debug: task_loss = {task_loss.item():.6f}")

            # if use_ts_distill:
            #     kl_loss = kl_criterion(output, target_ts)
            #     print(f"Debug: kl_loss = {kl_loss.item():.6f}")

            # # 初始化总损失
            # total_loss = task_loss
            # print(f"Debug: initial total_loss = {total_loss.item():.6f}")

            # if use_ts_distill:
            #     total_loss += kl_loss * args['train']['ts_loss_weight']
            #     print(f"Debug: after adding kl_loss, total_loss = {total_loss.item():.6f}")

            # # 如果启用隐藏层蒸馏，计算隐藏层损失
            # if use_hidden_distill:
            #     hidden_loss = hidden_criterion(teacher_hidden, student_hidden)
            #     print(f"Debug: hidden_loss = {hidden_loss.item():.6f}")
            #     print(f"Debug: before adding hidden_loss, total_loss = {total_loss.item():.6f}")
            #     total_loss += hidden_loss
            #     print(f"Debug: after adding hidden_loss, total_loss = {total_loss.item():.6f}")

            # 计算各项损失
            task_loss = criterion(output, target)
            if use_ts_distill:
                kl_loss = kl_criterion(output, target_ts)
            
            # 初始化总损失
            total_loss = task_loss.clone()

            if use_ts_distill:
                total_loss += kl_loss * args['train']['ts_loss_weight']
            
            # 如果启用隐藏层蒸馏，计算隐藏层损失
            if use_hidden_distill:
                hidden_loss = hidden_criterion(teacher_hidden, student_hidden)
                total_loss += hidden_loss
                
            # 如果启用注意力图蒸馏，计算注意力图损失
            if use_attn_distill:
                attn_loss = attn_criterion(teacher_attns, student_attns)
                total_loss += attn_loss
            
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss.append(total_loss.item())
            step_count += 1

            # 记录训练日志
            if step_count % args['result']['log_interval'] == 0:
                    # 对特定批次的样本进行详细分析
                logger.info("====== Feature Statistics Analysis ======")
                
                # 分析隐藏状态
                if use_hidden_distill:
                    t_hidden_norms = []
                    s_hidden_norms = []
                    t_hidden_means = []
                    s_hidden_means = []
                    diff_norms = []
                    
                    for i in range(len(teacher_hidden)):
                        t_norm = torch.norm(teacher_hidden[i]).item()
                        s_norm = torch.norm(student_hidden[i]).item()
                        t_mean = torch.mean(teacher_hidden[i]).item()
                        s_mean = torch.mean(student_hidden[i]).item()
                        diff = torch.norm(teacher_hidden[i] - student_hidden[i]).item()
                        
                        t_hidden_norms.append(t_norm)
                        s_hidden_norms.append(s_norm)
                        t_hidden_means.append(t_mean)
                        s_hidden_means.append(s_mean)
                        diff_norms.append(diff)
                        
                        logger.info(f"Layer {i} - Teacher hidden norm: {t_norm:.4f}, mean: {t_mean:.4f}")
                        logger.info(f"Layer {i} - Student hidden norm: {s_norm:.4f}, mean: {s_mean:.4f}")
                        logger.info(f"Layer {i} - Difference norm: {diff:.4f}")
                        
                        # 详细分析每层的MSE损失
                        layer_mse = F.mse_loss(teacher_hidden[i], student_hidden[i]).item()
                        logger.info(f"Layer {i} - MSE loss: {layer_mse:.6f}")
                    
                    logger.info(f"Teacher hidden - Avg norm: {np.mean(t_hidden_norms):.4f}, Avg mean: {np.mean(t_hidden_means):.4f}")
                    logger.info(f"Student hidden - Avg norm: {np.mean(s_hidden_norms):.4f}, Avg mean: {np.mean(s_hidden_means):.4f}")
                    logger.info(f"Average difference norm: {np.mean(diff_norms):.4f}")
                
                # 分析注意力图
                if use_attn_distill:
                    t_attn_norms = []
                    s_attn_norms = []
                    t_attn_means = []
                    s_attn_means = []
                    attn_diff_norms = []
                    
                    for i in range(len(teacher_attns)):
                        # 提取形状信息用于调试
                        t_shape = str(teacher_attns[i].shape)
                        s_shape = str(student_attns[i].shape)
                        logger.info(f"Layer {i} - Teacher attention shape: {t_shape}, Student attention shape: {s_shape}")
                        
                        t_attn_norm = torch.norm(teacher_attns[i]).item()
                        s_attn_norm = torch.norm(student_attns[i]).item()
                        t_attn_mean = torch.mean(teacher_attns[i]).item()
                        s_attn_mean = torch.mean(student_attns[i]).item()
                        
                        if teacher_attns[i].shape == student_attns[i].shape:
                            attn_diff = torch.norm(teacher_attns[i] - student_attns[i]).item()
                            layer_attn_mse = F.mse_loss(teacher_attns[i], student_attns[i]).item()
                        else:
                            attn_diff = float('nan')
                            layer_attn_mse = float('nan')
                            logger.warning(f"Layer {i} - Attention shapes don't match! Cannot compute difference.")
                        
                        t_attn_norms.append(t_attn_norm)
                        s_attn_norms.append(s_attn_norm)
                        t_attn_means.append(t_attn_mean)
                        s_attn_means.append(s_attn_mean)
                        attn_diff_norms.append(attn_diff)
                        
                        logger.info(f"Layer {i} - Teacher attention norm: {t_attn_norm:.6f}, mean: {t_attn_mean:.6f}")
                        logger.info(f"Layer {i} - Student attention norm: {s_attn_norm:.6f}, mean: {s_attn_mean:.6f}")
                        logger.info(f"Layer {i} - Difference norm: {attn_diff:.6f}")
                        logger.info(f"Layer {i} - Attention MSE loss: {layer_attn_mse:.8f}")
                    
                    logger.info(f"Teacher attention - Avg norm: {np.mean(t_attn_norms):.6f}, Avg mean: {np.mean(t_attn_means):.6f}")
                    logger.info(f"Student attention - Avg norm: {np.mean(s_attn_norms):.6f}, Avg mean: {np.mean(s_attn_means):.6f}")
                    valid_diffs = [d for d in attn_diff_norms if not np.isnan(d)]
                    if valid_diffs:
                        logger.info(f"Average attention difference norm: {np.mean(valid_diffs):.6f}")


                lr = optimizer.param_groups[0]['lr']
                log_message = f'epoch: {epoch_count}, step: {step_count}/{total_steps}, lr:{lr:.6f}, train_loss:{total_loss.item():.4f}'
                
                # 添加各项损失的详细信息
                log_message += f', task_loss:{task_loss.item():.4f}'
                if use_ts_distill:
                    log_message += f', kl_loss:{kl_loss.item():.4f}'
                if use_hidden_distill:
                    log_message += f', hidden_loss:{hidden_loss.item():.4f}'
                if use_attn_distill:
                    log_message += f', attn_loss:{attn_loss.item():.4f}'
                    
                logger.info(log_message)
                
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
                if use_hidden_distill and use_attn_distill:
                    output, target_ts, _, _ = model(input)
                elif use_hidden_distill or use_attn_distill:
                    output, target_ts, _ = model(input)
                else:
                    output, target_ts = model(input)
                task_loss = criterion(output, target)
                if use_ts_distill:
                    kl_loss = kl_criterion(output, target_ts)
                test_loss.append(task_loss.item())

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