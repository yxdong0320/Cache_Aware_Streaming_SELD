export CUDA_VISIBLE_DEVICES=3

# python test_A_dual_NS_to_S.py \
#         -c Test_Dual_Cache_APC_RC_24h_Chunk[100,49]_L8EM256_Fustep10_APC_NS_to_S \
#         -m results/Dual_Cache_APC_RC_24h_Chunk[100,49]_L8EM256_Fustep10_APC/checkpoints/checkpoint_epoch165_step90090.pth

# nohup python finetune_A_cache_chunk_limited_TS_hidstate_att_loss.py \
#         -c Finetune_T0.40_Cache_RC_24h_Chunk[100,4]_8ConformerLayer_SRD_logits_loss \
#         > train_logs/Finetune_T0.40_Cache_RC_24h_Chunk[100,4]_8ConformerLayer_SRD_logits_loss.log &

nohup python train_A_cache_chunk_limited_TS_hidstate_att_srd_loss.py \
        -c Train_T0.40_Cache_RC_24h_Chunk[100,4]_8ConformerLayer_srd_loss \
        > train_logs/Train_T0.40_Cache_RC_24h_Chunk[100,4]_8ConformerLayer_srd_loss.log &

# nohup python train_A_dual_mode_cache_CrossModalAPC.py \
#         -c Dual_Cache_APC_RC_24h_Chunk[100,49]_L8EM256_Fustep10_APC_tristage_weight \
#         > train_logs/Dual_Cache_APC_RC_24h_Chunk[100,49]_L8EM256_Fustep10_APC_tristage_weight.log &

# python finetune_A_cache_chunk_limited_TS_hidstate_att_loss.py \
#         -c Finetune_T0.40_Cache_RC_24h_Chunk[100,4]_8ConformerLayer_hiddenstate_loss \

# python finetune_A_cache_chunk_limited_TS_hidstate_att_loss_checkloss.py \
#         -c Finetune_T0.40_Cache_RC_24h_Chunk[100,4]_8ConformerLayer_hiddenstate_loss_check \

# python train_A_offline_APC_twostage.py -c Offline_RC_24h_8ConformerLayer_APC_fusteps_20_pretrain_m5 --apc_only

# python train_A_offline_APC_twostage.py -c Offline_RC_24h_8ConformerLayer_APC_fusteps_20_SELD_finetune_m5

# nohup python finetune_A_cache_chunk_limited_TS_hidstate_att_loss.py \
#         -c Finetune_TS_T0.40_Cache_RC_24h_Chunk[100,9]_8ConformerLayer_att_loss_hidstate_loss \
#         > train_logs/Finetune_TS_T0.40_Cache_RC_24h_Chunk[100,9]_8ConformerLayer_att_loss10_hidstate_loss0001.log &

# nohup python train_A_offline_APC_twostage.py \
#          -c Offline_RC_24h_8ConformerLayer_APC_fusteps_20_SELD_finetune_m4 \
#          > train_logs/Offline_RC_24h_8ConformerLayer_APC_fusteps_20_SELD_finetune_m4.log &

# export CUDA_VISIBLE_DEVICES=1
# nohup python train_A_offline_APC_twostage.py \
#         -c Offline_RC_24h_8ConformerLayer_APC_fusteps_20_SELD_finetune \
#         > train_logs/Offline_RC_24h_8ConformerLayer_APC_fusteps_20_SELD_finetune.log &

# export CUDA_VISIBLE_DEVICES=3
# nohup python train_A_offline_HPC.py \
#          -c Offline_RC_24h_8ConformerLayer_HPC_fusteps_20 \
#          > train_logs/Offline_RC_24h_8ConformerLayer_HPC_fusteps_20_hpcloss0005_2.log &

# export CUDA_VISIBLE_DEVICES=0
# nohup python train_A_cache_chunk_limited.py \
#     -c Cache_RC_24h_Chunk[100,49]_8ConformerLayer_3s \
#     > train_logs/Cache_RC_24h_Chunk[100,49]_8ConformerLayer_3s.log &

# export CUDA_VISIBLE_DEVICES=2
# nohup python train_A_offline.py \
#     -c Offline_RC_24h_8ConformerLayer_1s \
#     > train_logs/Offline_RC_24h_8ConformerLayer_1s.log &    

# export CUDA_VISIBLE_DEVICES=3
# nohup python train_A_offline_APC.py \
#     -c Offline_RC_24h_8ConformerLayer_APC_fusteps_20 \
#     > train_logs/Offline_RC_24h_8ConformerLayer_APC_fusteps_20.log &   

# export CUDA_VISIBLE_DEVICES=0
# nohup python train_A_cache_chunk_limited_APC.py \
#     -c Cache_RC_APC_24h_Chunk[100,49]_8ConformerLayer_fusteps_20 \
#     > train_logs/Cache_RC_APC_24h_Chunk[100,49]_8ConformerLayer_fusteps_20.log &   

# export CUDA_VISIBLE_DEVICES=2
# nohup python train_A_dual_mode_cache.py \
#     -c Dual_Cache_RC_24h_Chunk[100,4]_L8EM256 \
#     > train_logs/Dual_Cache_RC_24h_Chunk[100,4]_L8EM256.log &    

# export CUDA_VISIBLE_DEVICES=1
# nohup python train_A_dual_mode_cache_APC.py \
#     -c Dual_Cache_APC_RC_24h_Chunk[100,4]_L8EM256_Fustep20 \
#     > train_logs/Dual_Cache_APC_RC_24h_Chunk[100,4]_L8EM256_Fustep20.log &    

# export CUDA_VISIBLE_DEVICES=3
# nohup python finetune_A_cache_chunk_limited_TS.py \
#     -c Finetune_TS_T0.40_Cache_RC_24h_Chunk[100,4]_L8EM256 \
#     > train_logs/Finetune_TS_T0.40_Cache_RC_24h_Chunk[100,4]_L8EM256.log &