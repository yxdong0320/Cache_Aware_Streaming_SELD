import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import os
from torch.utils.data import DataLoader

def load_nonstream_to_stream_model(pretrained_model_path, target_model, device='cuda'):
    """
    将预训练的双模态模型中的非流式参数加载到流式模型中
    
    Args:
        pretrained_model_path: 预训练模型路径
        target_model: 目标流式模型
        device: 设备
    
    Returns:
        加载成功的参数数量统计
    """
    print(f"开始从 {pretrained_model_path} 加载参数...")
    
    # 加载预训练权重
    pretrained_dict = torch.load(pretrained_model_path, map_location=device)
    target_dict = target_model.state_dict()
    
    loaded_params = []
    skipped_params = []
    shape_mismatched = []
    
    print("开始参数映射...")
    
    for pretrained_key, pretrained_param in pretrained_dict.items():
        target_key = None
        should_load = False
        
        # 1. 直接映射的共享参数
        if any(pretrained_key.startswith(shared) for shared in 
               ['input_projection', 'sed_out_layer', 'out_layer']):
            target_key = pretrained_key
            should_load = True
            
        # 2. ResNet参数映射: 非流式 → 流式
        elif pretrained_key.startswith('resnet.'):
            if 'nonstream_conv1' in pretrained_key:
                target_key = pretrained_key.replace('nonstream_conv1', 'stream_conv1')
                should_load = True
            elif 'nonstream_conv' in pretrained_key and 'layer' in pretrained_key:
                target_key = pretrained_key.replace('nonstream_conv', 'stream_conv')
                should_load = True
            # 共享的层(ln1, ln2, downsample, conv5等)直接映射
            elif any(shared in pretrained_key for shared in 
                    ['ln1', 'ln2', 'downsample', 'conv5', 'maxpool']):
                target_key = pretrained_key
                should_load = True
                
        # 3. Conformer参数映射
        elif pretrained_key.startswith('conformer_layers.'):
            # 前馈网络参数映射（非流式 → 流式）
            if 'nonstream_ff1' in pretrained_key:
                target_key = pretrained_key.replace('nonstream_ff1', 'stream_ff1')
                should_load = True
            elif 'nonstream_ff2' in pretrained_key:
                target_key = pretrained_key.replace('nonstream_ff2', 'stream_ff2')
                should_load = True
                
            # 注意力层参数映射: NonStreamingAttention → CausalAttention
            elif 'nonstream_attn' in pretrained_key:
                target_key = pretrained_key.replace('nonstream_attn', 'stream_attn')
                should_load = True
                
            # 卷积层参数映射: NonStreamingConformerConvModule → ConformerConvModule
            elif 'nonstream_conv' in pretrained_key:
                target_key = pretrained_key.replace('nonstream_conv', 'stream_conv')
                should_load = True
                
            # 共享的归一化层和缩放层
            elif any(shared in pretrained_key for shared in 
                    ['attn_norm', 'ff1_norm', 'ff2_norm', 'post_norm', 'scale_ff1', 'scale_ff2']):
                target_key = pretrained_key
                should_load = True
        
        # 4. APC相关参数
        elif any(pretrained_key.startswith(apc_prefix) for apc_prefix in 
                ['cross_modal_apc', 'nonstream_apc', 'stream_apc']):
            if pretrained_key in target_dict:
                target_key = pretrained_key
                should_load = True
        
        # 执行参数加载
        if should_load and target_key and target_key in target_dict:
            if pretrained_param.shape == target_dict[target_key].shape:
                target_dict[target_key].copy_(pretrained_param)
                loaded_params.append(f"{pretrained_key} -> {target_key}")
            else:
                shape_mismatched.append(
                    f"{pretrained_key} {pretrained_param.shape} vs {target_key} {target_dict[target_key].shape}"
                )
        else:
            if should_load and target_key:
                skipped_params.append(f"{pretrained_key} -> {target_key} (not found in target)")
            else:
                skipped_params.append(pretrained_key)
    
    # 加载到目标模型
    target_model.load_state_dict(target_dict)
    
    # 打印统计信息
    print(f"\n{'='*60}")
    print(f"参数加载完成!")
    print(f"成功加载参数数量: {len(loaded_params)}")
    print(f"跳过参数数量: {len(skipped_params)}")
    print(f"形状不匹配数量: {len(shape_mismatched)}")
    
    if loaded_params:
        print(f"\n成功加载的参数 (前10个):")
        for param in loaded_params[:10]:
            print(f"  ✓ {param}")
        if len(loaded_params) > 10:
            print(f"  ... 还有 {len(loaded_params)-10} 个参数")
    
    if shape_mismatched:
        print(f"\n形状不匹配的参数:")
        for param in shape_mismatched[:5]:
            print(f"  ✗ {param}")
        if len(shape_mismatched) > 5:
            print(f"  ... 还有 {len(shape_mismatched)-5} 个参数")
            
    print(f"{'='*60}\n")
    
    return len(loaded_params), len(skipped_params), len(shape_mismatched)

def create_stream_only_model_from_dual_mode(dual_mode_checkpoint, config_args, logger=None):
    """
    从双模态检查点创建纯流式模型
    """
    if logger:
        logger.info("从双模态模型创建纯流式模型...")
    else:
        print("从双模态模型创建纯流式模型...")
    
    # 创建目标流式模型
    from models.dual_mode_cache_resnet_conformer_APC import DualModeResnetConformerCrossModalAPC
    
    stream_model = DualModeResnetConformerCrossModalAPC(
        in_channel=config_args['model']['in_channel'],
        in_dim=config_args['model']['in_dim'], 
        out_dim=config_args['model']['out_dim'],
        att_context_size=config_args['model']['att_context_size'],
        num_conformer_layer=config_args['model']['num_conformer_layer'],
        encoder_dim=config_args['model']['encoder_dim'],
        apc_future_steps=config_args['model']['apc_future_steps'],
        use_nonstream_apc=False  # 纯流式模型不需要非流式APC
    )
    
    # 加载参数
    loaded_count, skipped_count, mismatch_count = load_nonstream_to_stream_model(
        dual_mode_checkpoint, 
        stream_model
    )
    
    if logger:
        logger.info(f"参数加载统计: 成功={loaded_count}, 跳过={skipped_count}, 不匹配={mismatch_count}")
    
    return stream_model, loaded_count, skipped_count, mismatch_count

def test_stream_model_performance(stream_model, test_dataloader, criterion, device, logger=None):
    """
    测试纯流式模型性能
    """
    from utils.sed_doa import SedDoaResult
    from utils.cls_tools.cls_compute_seld_results import ComputeSELDResults
    from utils.write_csv import write_output_format_file
    import time
    
    if logger:
        logger.info("开始测试流式模型性能...")
    else:
        print("开始测试流式模型性能...")
    
    stream_model.eval()
    stream_model = stream_model.to(device)
    
    test_stream_loss = []
    test_result_stream = SedDoaResult(segment_length=test_dataloader.dataset.segment_len)
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, data in enumerate(test_dataloader):
            input_data = data['input'].to(device)
            target = data['target'].to(device)
            
            # 流式模式推理
            stream_pred = stream_model(input_data, mode='streaming')
            stream_loss = criterion(stream_pred, target)
            test_stream_loss.append(stream_loss.item())
            
            # 收集结果用于SELD评估
            test_result_stream.add_items(data['wav_names'], stream_pred)
            
            if (batch_idx + 1) % 10 == 0:
                if logger:
                    logger.info(f"已处理 {batch_idx + 1}/{len(test_dataloader)} 批次")
                else:
                    print(f"已处理 {batch_idx + 1}/{len(test_dataloader)} 批次")
    
    test_time = time.time() - start_time
    avg_test_loss = np.mean(test_stream_loss)
    
    # 获取预测结果
    output_dict_stream = test_result_stream.get_result()
    
    if logger:
        logger.info(f"流式模型测试完成，耗时: {test_time:.2f}s, 平均损失: {avg_test_loss:.4f}")
    else:
        print(f"流式模型测试完成，耗时: {test_time:.2f}s, 平均损失: {avg_test_loss:.4f}")
    
    return output_dict_stream, avg_test_loss, test_time

def evaluate_and_save_results(output_dict, args, model_name, logger=None):
    """
    评估结果并保存CSV文件
    """
    from utils.cls_tools.cls_compute_seld_results import ComputeSELDResults
    from utils.write_csv import write_output_format_file
    
    # 保存CSV文件
    dcase_output_dir = os.path.join(args['result']['dcase_output_dir'], f'{model_name}_test')
    os.makedirs(dcase_output_dir, exist_ok=True)
    
    for csv_name, perfile_out_dict in output_dict.items():
        output_file = os.path.join(dcase_output_dir, f'{csv_name}.csv')
        write_output_format_file(output_file, perfile_out_dict)
    
    # SELD评估
    score_obj = ComputeSELDResults(ref_files_folder=args['data']['ref_files_dir'])
    ER, F, LE, LR, seld_scr, classwise_scr = score_obj.get_SELD_Results(dcase_output_dir)
    
    if logger:
        logger.info(f'{model_name} ER/F/LE/LR/SELD: {ER:.4f}/{F:.4f}/{LE:.4f}/{LR:.4f}/{seld_scr:.4f}')
    else:
        print(f'{model_name} ER/F/LE/LR/SELD: {ER:.4f}/{F:.4f}/{LE:.4f}/{LR:.4f}/{seld_scr:.4f}')
    
    return ER, F, LE, LR, seld_scr, dcase_output_dir

def main_test_stream_model(args, dual_mode_checkpoint_path):
    """
    主测试函数：从双模态模型初始化流式模型并测试性能
    """
    # 设置日志
    log_output_folder = os.path.dirname(args['result']['log_output_path'])
    os.makedirs(log_output_folder, exist_ok=True)
    
    # 修改日志文件名以区分
    log_file = args['result']['log_output_path'].replace('.log', '_stream_test.log')
    logging.basicConfig(
        filename=log_file, 
        filemode='w', 
        level=logging.INFO, 
        format='%(levelname)s: %(asctime)s: %(message)s', 
        datefmt='%m/%d/%Y %H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    logger.info("开始流式模型测试")
    logger.info(f"配置参数: {args}")
    logger.info(f"双模态检查点路径: {dual_mode_checkpoint_path}")
    
    # 数据加载
    from lmdb_data_loader_A import LmdbDataset
    from utils.sed_doa import SedDoaLoss, process_foa_input_sed_doa_labels
    
    test_split = [4]
    test_dataset = LmdbDataset(
        args['data']['test_lmdb_dir'], 
        test_split, 
        normalized_features_wts_file=args['data']['norm_file'],
        ignore=args['data']['test_ignore'], 
        segment_len=args['data']['segment_len'], 
        data_process_fn=process_foa_input_sed_doa_labels
    )
    test_dataloader = DataLoader(
        dataset=test_dataset, 
        batch_size=args['data']['batch_size'], 
        shuffle=False, 
        num_workers=args['train']['test_num_workers'], 
        collate_fn=test_dataset.collater
    )
    
    # 损失函数
    criterion = SedDoaLoss(loss_weight=[0.1, 1])
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 1. 创建并初始化流式模型
    stream_model, loaded_count, skipped_count, mismatch_count = create_stream_only_model_from_dual_mode(
        dual_mode_checkpoint_path, args, logger
    )
    
    # 2. 测试流式模型性能
    stream_output_dict, stream_loss, stream_test_time = test_stream_model_performance(
        stream_model, test_dataloader, criterion, device, logger
    )
    
    # 3. 评估并保存结果
    stream_ER, stream_F, stream_LE, stream_LR, stream_seld_scr, stream_output_dir = evaluate_and_save_results(
        stream_output_dict, args, 'stream_from_dual', logger
    )
    
    # 4. 总结报告
    logger.info("="*80)
    logger.info("流式模型测试报告")
    logger.info(f"双模态检查点: {dual_mode_checkpoint_path}")
    logger.info(f"参数加载统计: 成功={loaded_count}, 跳过={skipped_count}, 不匹配={mismatch_count}")
    logger.info(f"测试时间: {stream_test_time:.2f}s")
    logger.info(f"平均损失: {stream_loss:.4f}")
    logger.info(f"SELD指标: ER={stream_ER:.4f}, F={stream_F:.4f}, LE={stream_LE:.4f}, LR={stream_LR:.4f}, SELD={stream_seld_scr:.4f}")
    logger.info(f"结果保存路径: {stream_output_dir}")
    logger.info("="*80)
    
    return {
        'loaded_params': loaded_count,
        'skipped_params': skipped_count,
        'mismatch_params': mismatch_count,
        'test_time': stream_test_time,
        'test_loss': stream_loss,
        'ER': stream_ER,
        'F': stream_F,
        'LE': stream_LE,
        'LR': stream_LR,
        'SELD': stream_seld_scr,
        'output_dir': stream_output_dir
    }

# 使用示例
if __name__ == "__main__":
    import argparse
    import yaml
    
    # 解析命令行参数
    parser = argparse.ArgumentParser('test_stream_from_dual')
    parser.add_argument('-c', '--config_name', type=str, 
                       default='foa_dev_multi_accdoa_nopool', help='name of config')
    parser.add_argument('-m', '--model_path', type=str, required=True,
                       help='path to dual mode checkpoint')
    input_args = parser.parse_args()
    
    # 加载配置
    with open(os.path.join('config', f'{input_args.config_name}.yaml'), 'r') as f:
        args = yaml.safe_load(f)
    
    # 运行测试
    results = main_test_stream_model(args, input_args.model_path)
    
    print("\n测试完成!")
    print(f"SELD分数: {results['SELD']:.4f}")
    print(f"详细结果请查看日志文件")