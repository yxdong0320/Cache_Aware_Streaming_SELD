import torch
import thop
import torchinfo
import sys
import os
from copy import deepcopy
import matplotlib.pyplot as plt

# Assuming the model modules are in the same directory
sys.path.append('.')
from cache_resnet_conformer_change_cache_len import ResnetConformer_sed_doa_nopool

def analyze_model_complexity():
    # Initialize model with the same parameters as in the main function
    att_context_size = [100, 49]
    model = ResnetConformer_sed_doa_nopool(
        in_channel=7, 
        in_dim=64, 
        out_dim=39, 
        att_context_size=att_context_size,
        num_conformer_layer=8,
        encoder_dim=256,
        inference_cache_len= 1
    )
    model.eval()
    
    # Get batch size and chunk size from the main function
    batch_size = 1
    chunk_size = att_context_size[1] + 1  # 25 frames
    
    # Model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "="*80)
    print(f"PARAMETER COUNT:")
    print(f"Total parameters: {total_params/1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params/1e6:.2f}M")
    print("="*80)
    
    # Create inputs for initial and subsequent inference
    initial_input = torch.randn(batch_size, 7, chunk_size, 64)
    
    # 1. Initial Inference (first chunk)
    print("\nANALYZING INITIAL INFERENCE (FIRST CHUNK)")
    print("="*80)
    
    # Initialize caches
    resnet_cache = model.get_initial_cache_resnet(batch_size)
    conformer_cache = model.get_initial_cache_conformer(batch_size)
    
    # Calculate FLOPs for initial inference using thop
    macs_initial, params_initial = thop.profile(
        deepcopy(model), 
        inputs=(initial_input, resnet_cache, conformer_cache),
        verbose=False
    )
    
    print(f"Initial inference MACs: {macs_initial/1e9:.4f} G")
    print(f"Initial inference parameters: {params_initial/1e6:.2f} M")
    
    # 2. Subsequent Inference (with cached context)
    print("\nANALYZING SUBSEQUENT INFERENCE (WITH CACHE)")
    print("="*80)
    
    # Get updated caches by running the model once
    with torch.no_grad():
        _, (next_resnet_cache, next_conformer_cache) = model(
            initial_input, 
            resnet_cache=resnet_cache, 
            conformer_cache=conformer_cache
        )
    
    # Now profile with the updated caches
    subsequent_input = torch.randn(batch_size, 7, chunk_size, 64)
    
    macs_subsequent, params_subsequent = thop.profile(
        deepcopy(model), 
        inputs=(subsequent_input, next_resnet_cache, next_conformer_cache),
        verbose=False
    )
    
    print(f"Subsequent inference MACs: {macs_subsequent/1e9:.4f} G")
    print(f"Subsequent inference parameters: {params_subsequent/1e6:.2f} M")
    
    # Additional model summary using torchinfo
    print("\nMODEL SUMMARY FROM TORCHINFO")
    print("="*80)
    summary = torchinfo.summary(
        model, 
        input_data=[initial_input, resnet_cache, conformer_cache],
        depth=3,
        verbose=0
    )
    
    # Compare initial vs subsequent
    print("\nCOMPARISON")
    print("="*80)
    print(f"Initial vs Subsequent MACs difference: {(macs_initial-macs_subsequent)/1e9:.4f} G")
    print(f"Initial requires {macs_initial/macs_subsequent:.2f}x more computation than subsequent")

if __name__ == "__main__":
    analyze_model_complexity()