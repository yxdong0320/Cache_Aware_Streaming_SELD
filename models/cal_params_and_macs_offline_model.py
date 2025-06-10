import torch
import thop
import torchinfo
import sys
import os
from copy import deepcopy

sys.path.append('.')
from resnet_conformer_audio import ResnetConformer_sed_doa_nopool

def analyze_non_streaming_model():
    # Initialize model with the provided parameters
    model = ResnetConformer_sed_doa_nopool(in_channel=7, in_dim=64, out_dim=39)
    model.eval()
    
    # Create input tensor as specified
    input_tensor = torch.randn(1, 7, 250, 64)
    
    # Calculate parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "="*80)
    print("NON-STREAMING MODEL PARAMETER COUNT:")
    print(f"Total parameters: {total_params/1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params/1e6:.2f}M")
    print("="*80)
    
    # Calculate FLOPs using thop
    macs, params = thop.profile(
        model, 
        inputs=(input_tensor,),
        verbose=False
    )
    
    print(f"\nMACs: {macs/1e9:.4f} G")
    print(f"Parameters measured by thop: {params/1e6:.2f} M")
    
    # Additional model summary using torchinfo
    print("\nMODEL SUMMARY FROM TORCHINFO")
    print("="*80)
    summary = torchinfo.summary(
        model, 
        input_data=[input_tensor],
        depth=3,
        verbose=0
    )
    
    # Calculate throughput estimation
    batch_size = 1
    if torch.cuda.is_available():
        model = model.cuda()
        input_tensor = input_tensor.cuda()
        
        # Warm-up
        for _ in range(10):
            with torch.no_grad():
                _ = model(input_tensor)
        
        # Measure
        import time
        torch.cuda.synchronize()
        start = time.time()
        iterations = 100
        for _ in range(iterations):
            with torch.no_grad():
                _ = model(input_tensor)
        torch.cuda.synchronize()
        end = time.time()
        
        inference_time = (end - start) / iterations
        print(f"\nAverage inference time per batch: {inference_time*1000:.2f} ms")
        print(f"Throughput: {batch_size/inference_time:.2f} samples/sec")
    else:
        print("\nCUDA not available for throughput testing. Running on CPU only.")
    
    print("\n" + "="*80)
    print("COMPONENT-WISE BREAKDOWN")
    print("="*80)
    
    # Analyze ResNet part
    resnet_macs, _ = thop.profile(
        model.resnet, 
        inputs=(input_tensor,),
        verbose=False
    )
    print(f"ResNet MACs: {resnet_macs/1e9:.4f} G ({resnet_macs/macs*100:.2f}% of total)")
    
    # Create intermediate input for conformer layers
    with torch.no_grad():
        conv_outputs = model.resnet(input_tensor)
        N, C, T, W = conv_outputs.shape
        conv_outputs = conv_outputs.permute(0, 2, 1, 3).reshape(N, T, C*W)
        conformer_input = model.input_projection(conv_outputs)
    
    # Analyze one conformer layer
    one_layer_macs, _ = thop.profile(
        model.conformer_layers[0], 
        inputs=(conformer_input,),
        verbose=False
    )
    print(f"Single Conformer Layer MACs: {one_layer_macs/1e9:.4f} G")
    print(f"All 8 Conformer Layers MACs (estimated): {(one_layer_macs*8)/1e9:.4f} G ({one_layer_macs*8/macs*100:.2f}% of total)")
    
    # Analyze output layers
    with torch.no_grad():
        for layer in model.conformer_layers:
            conformer_input = layer(conformer_input)
        outputs = conformer_input.permute(0, 2, 1)
        outputs = model.t_pooling(outputs)
        outputs = outputs.permute(0, 2, 1)
    
    sed_macs, _ = thop.profile(
        model.sed_out_layer, 
        inputs=(outputs,),
        verbose=False
    )
    
    doa_macs, _ = thop.profile(
        model.out_layer, 
        inputs=(outputs,),
        verbose=False
    )
    
    print(f"SED layer MACs: {sed_macs/1e9:.4f} G ({sed_macs/macs*100:.2f}% of total)")
    print(f"DOA layer MACs: {doa_macs/1e9:.4f} G ({doa_macs/macs*100:.2f}% of total)")

if __name__ == "__main__":
    analyze_non_streaming_model()