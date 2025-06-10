# 流式推理示例（块级别处理）
def streaming_inference(model, input_tensor, chunk_size, device):
    # 设置为评估模式
    model.eval()
    
    batch_size = input_tensor.shape[0]
    
    # 初始化缓存
    resnet_cache = model.get_initial_cache_resnet(batch_size)
    conformer_cache = model.get_initial_cache_conformer(batch_size)
    
    # 存储所有输出
    all_outputs = []
    
    # 流式推理
    with torch.no_grad():
        for i in range(0, input_tensor.shape[2], chunk_size):
            chunk = input_tensor[:, :, i:i+chunk_size, :].to(device)
            
            # 推理并更新缓存
            output, (resnet_cache, conformer_cache) = model(
                chunk, 
                mode='streaming',
                resnet_cache=resnet_cache, 
                conformer_cache=conformer_cache
            )
            
            all_outputs.append(output)
    
    # 拼接所有输出
    return torch.cat(all_outputs, dim=1)