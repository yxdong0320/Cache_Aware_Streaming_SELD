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

from models.cache_resnet_conformer_TS import ResnetConformer_sed_doa_nopool_TS_hidstate_att_loss, ResnetConformer_sed_doa_nopool_TS_hidstate_att_srd_loss
from models.cache_resnet_conformer_TS import ResnetConformer_sed_doa_nopool_Cache_TS_hidstate_att_loss
from lr_scheduler.tri_stage_lr_scheduler import TriStageLRScheduler
from utils.cls_tools.cls_compute_seld_results import ComputeSELDResults
from utils.write_csv import write_output_format_file
from utils.sed_doa import SedDoaResult, process_foa_input_sed_doa_labels, SedDoaLoss, SedDoaKLLoss_2
from utils.sed_doa import HiddenStateMSELoss, AttentionMapMSELoss, HiddenStateMSELoss_weighted, AttentionMapMSELoss_weighted
from utils.sed_doa import HiddenStateMSELoss_norm, SimpleAttentionDivergenceLoss, HiddenStateCosineLoss
from utils.sed_doa import SemanticRepresentationDistillationLoss_KLLoss_2


def check_parameter_compatibility(teacher_dict, student_dict, model_name="模型"):
    """检查教师模型参数与学生模型的兼容性"""
    compatible_keys = []
    incompatible_keys = []
    missing_keys = []
    
    print(f"\n=== {model_name}参数兼容性检测 ===")
    
    # 检查教师模型参数在学生模型中的兼容性
    for key in teacher_dict.keys():
        if key in student_dict:
            teacher_shape = teacher_dict[key].shape
            student_shape = student_dict[key].shape
            
            if teacher_shape == student_shape:
                compatible_keys.append(key)
            else:
                incompatible_keys.append((key, teacher_shape, student_shape))
                print(f"⚠️  参数形状不匹配: {key}")
                print(f"   教师模型: {teacher_shape}, 学生模型: {student_shape}")
        else:
            missing_keys.append(key)
    
    # 打印统计信息
    print(f"✅ 兼容参数: {len(compatible_keys)}")
    print(f"⚠️  形状不匹配: {len(incompatible_keys)}")
    print(f"❌ 缺失参数: {len(missing_keys)}")
    
    if missing_keys:
        print(f"缺失的参数键: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
    
    return compatible_keys, incompatible_keys, missing_keys


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
    use_srd_distill = args['train'].get('use_srd_distill', False)  # 新增SRD控制
    hidden_distill_weighted = args['train'].get('hidden_distill_weighted', False)
    att_distill_weighted = args['train'].get('att_distill_weighted', False)
    use_cache_as_T = args['model'].get('use_cache_as_T', False)

    criterion = SedDoaLoss(loss_weight=[0.1,1])
    # 添加隐藏层和注意力图的损失函数
    if use_hidden_distill and not hidden_distill_weighted: 
        hidden_criterion = HiddenStateMSELoss(loss_weight=args['train'].get('hidden_loss_weight', 0.2))
    elif use_hidden_distill and hidden_distill_weighted:
        hidden_distill_layer_weight = args['train'].get('hidden_distill_layer_weight', [1, 1, 1, 1, 1, 1, 1, 1])
        hidden_criterion = HiddenStateMSELoss_weighted(loss_weight=args['train'].get('hidden_loss_weight', 0.2),
                                                       layer_weights=hidden_distill_layer_weight)
        # hidden_criterion = HiddenStateMSELoss_norm(loss_weight=args['train'].get('hidden_loss_weight', 0.2))
    if use_attn_distill and not att_distill_weighted:
        attn_criterion = AttentionMapMSELoss(loss_weight=args['train'].get('attn_loss_weight', 0.05))
        # attn_criterion = SimpleAttentionDivergenceLoss(loss_weight=args['train'].get('attn_loss_weight', 0.05))
    elif use_attn_distill and att_distill_weighted:
        attn_distill_layer_weight = args['train'].get('attn_distill_layer_weight', [1, 1, 1, 1, 1, 1, 1, 1])
        attn_criterion = AttentionMapMSELoss_weighted(loss_weight=args['train'].get('attn_loss_weight', 0.05),
                                                      layer_weights=attn_distill_layer_weight)
        # attn_criterion = SimpleAttentionDivergenceLoss(loss_weight=args['train'].get('attn_loss_weight', 0.05))
    if use_ts_distill:
        kl_criterion = SedDoaKLLoss_2(loss_weight=[0.1, 1]) 
    if use_srd_distill:
        srd_criterion = SemanticRepresentationDistillationLoss_KLLoss_2(loss_weight=[0.1, 1])
        model = ResnetConformer_sed_doa_nopool_TS_hidstate_att_srd_loss(
            in_channel=args['model']['in_channel'], 
            in_dim=args['model']['in_dim'], 
            out_dim=args['model']['out_dim'],
            att_context_size=args['model']['att_context_size'],
            num_conformer_layer=args['model']['num_conformer_layer'],
            encoder_dim=args['model']['encoder_dim'],
            use_hidden_distill=use_hidden_distill,
            use_attn_distill=use_attn_distill,
            use_srd_distill=use_srd_distill,  # 新增参数
            )
    elif use_cache_as_T:
        model = ResnetConformer_sed_doa_nopool_Cache_TS_hidstate_att_loss(
                in_channel=args['model']['in_channel'], 
                in_dim=args['model']['in_dim'], 
                out_dim=args['model']['out_dim'],
                att_context_size=args['model']['att_context_size'],
                num_conformer_layer=args['model']['num_conformer_layer'],
                encoder_dim=args['model']['encoder_dim'],
                T_att_context_size = args['model']['T_att_context_size'],
                use_hidden_distill=use_hidden_distill,
                use_attn_distill=use_attn_distill,
                )
    else:
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

    if args['model']['pre-train']:
        model_dict = model.state_dict()
        model_dict_t = model.teacher_model.state_dict()
        pretrained_dict_t = torch.load(args['model']['pre-train_model_t'], map_location=device)

        s_load_t_params = args['model'].get('s_load_t_params', False)
        if s_load_t_params:
            # 如果s_load_t_params为True，则将教师模型的参数加载到学生模型中
            print(f'正在加载教师模型参数到学生模型...')
            pretrained_dict_s = torch.load(args['model']['pre-train_model_t'], map_location=device)
                    # **新增：检测教师模型参数与学生模型的兼容性**
            compatible_keys, incompatible_keys, missing_keys = check_parameter_compatibility(
                pretrained_dict_s, model_dict, "教师->学生"
            )
            
            # 如果有不兼容的参数，给出警告
            if incompatible_keys:
                print(f"\n⚠️  警告: 发现 {len(incompatible_keys)} 个参数形状不匹配，将跳过这些参数")
        else:
            pretrained_dict_s = torch.load(args['model']['pre-train_model_s'], map_location=device)
        

        # 加载学生模型时只加载存在的键
        key_list = [key for key in pretrained_dict_s.keys()]  
        for key in key_list:
            if key in model_dict:
                model_dict[key] = pretrained_dict_s[key]
        
        # 加载教师模型（保持原有逻辑）
        key_list_t = [key for key in pretrained_dict_t.keys()]
        if 'APC' in args['model']['pre-train_model_t']:
            filtered_key_list_t = [key for key in key_list_t if 'apc' not in key.lower()]
            print(f"检测到APC模型, 忽略了 {len(key_list_t) - len(filtered_key_list_t)} 个APC相关层")
            for key in filtered_key_list_t:
                if key in model_dict_t:
                    model_dict_t[key] = pretrained_dict_t[key]
                else:
                    print(f"警告: 目标模型中不存在层 {key}")
        else:
            for key in key_list_t:
                model_dict_t[key] = pretrained_dict_t[key]
        
        # 加载模型权重
        model.load_state_dict(model_dict, strict=False)
        model.teacher_model.load_state_dict(model_dict_t, strict=False)
        
        # **新增：feature_adapter恒等映射初始化**
        if use_srd_distill and hasattr(model, 'feature_adapter'):
            print("正在为feature_adapter应用恒等映射初始化...")
            linear_layers = [module for module in model.feature_adapter.modules() if isinstance(module, nn.Linear)]
            
            for i, module in enumerate(linear_layers):
                if i == 0:  # 第一个Linear层使用恒等映射
                    print(f"  Linear层 {i}: 恒等映射初始化 (weight shape: {module.weight.shape})")
                    nn.init.eye_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                else:  # 其他Linear层使用Xavier初始化
                    print(f"  Linear层 {i}: Xavier初始化 (weight shape: {module.weight.shape})")
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            print("feature_adapter恒等映射初始化完成")
        
        print('成功加载教师和学生模型')

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

            # **修改：根据模型配置获取不同的输出**
            model_outputs = model(input)
            output = model_outputs[0]
            target_ts = model_outputs[1]
            
            # **修改：解析模型输出**
            output_idx = 2
            teacher_hidden, student_hidden = None, None
            teacher_attns, student_attns = None, None
            student_cross_output, teacher_cross_output = None, None
            
            if use_hidden_distill:
                teacher_hidden, student_hidden = model_outputs[output_idx]
                output_idx += 1
                
            if use_attn_distill:
                teacher_attns, student_attns = model_outputs[output_idx]
                output_idx += 1
                
            # **新增：解析SRD输出**
            if use_srd_distill:
                student_cross_output, teacher_cross_output = model_outputs[output_idx]
                output_idx += 1

            # 计算各项损失
            task_loss = criterion(output, target)
            total_loss = task_loss.clone()

            if use_ts_distill:
                kl_loss = kl_criterion(output, target_ts)
                total_loss += kl_loss * args['train']['ts_loss_weight']
            
            # 如果启用隐藏层蒸馏，计算隐藏层损失
            if use_hidden_distill:
                hidden_loss = hidden_criterion(teacher_hidden, student_hidden)
                total_loss += hidden_loss
                
            # 如果启用注意力图蒸馏，计算注意力图损失
            if use_attn_distill:
                attn_loss = attn_criterion(teacher_attns, student_attns)
                total_loss += attn_loss
                
            # **新增：如果启用SRD蒸馏，计算SRD损失**
            if use_srd_distill:
                srd_loss = srd_criterion(student_cross_output, teacher_cross_output)
                total_loss += srd_loss * args['train']['srd_loss_weight']

            # # 根据模型配置获取不同的输出
            # if use_hidden_distill and use_attn_distill:
            #     output, target_ts, (teacher_hidden, student_hidden), (teacher_attns, student_attns) = model(input)
            # elif use_hidden_distill:
            #     output, target_ts, (teacher_hidden, student_hidden) = model(input)
            # elif use_attn_distill:
            #     output, target_ts, (teacher_attns, student_attns) = model(input)
            # else:
            #     output, target_ts = model(input)

            # pdb.set_trace()
            
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss.append(total_loss.item())
            step_count += 1

            # 记录训练日志
            if step_count % args['result']['log_interval'] == 0:
                # pdb.set_trace()
                lr = optimizer.param_groups[0]['lr']
                log_message = f'epoch: {epoch_count}, step: {step_count}/{total_steps}, lr:{lr:.6f}, train_loss:{total_loss.item():.4f}'
                
                # 添加各项损失的详细信息
                log_message += f', task_loss:{task_loss.item():.4f}' 
                if use_ts_distill:
                    log_message += f', ts_loss:{kl_loss.item():.4f}'
                if use_hidden_distill:
                    log_message += f', hidden_loss:{hidden_loss.item():.4f}'
                if use_attn_distill:
                    log_message += f', attn_loss:{attn_loss.item():.4f}'
                if use_srd_distill:
                    log_message += f', srd_loss:{srd_loss.item():.4f}'
                    
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
                model_outputs = model(input)
                output = model_outputs[0]
                target_ts = model_outputs[1]

                # if use_hidden_distill and use_attn_distill:
                #     output, target_ts, _, _ = model(input)
                # elif use_hidden_distill or use_attn_distill:
                #     output, target_ts, _ = model(input)
                # else:
                #     output, target_ts = model(input)
                task_loss = criterion(output, target)
                # kl_loss = kl_criterion(output, target_ts)
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
        logger.info('epoch: {}, step: {}/{}, train_time:{:.2f}, test_time:{:.2f}, average_train_total_loss:{:.4f}, average_test_task_loss:{:.4f}'.format(epoch_count, step_count, total_steps, train_time, test_time, np.mean(train_loss), np.mean(test_loss)))
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