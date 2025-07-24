# export CUDA_VISIBLE_DEVICES=0
# nohup python test_A_dual_cache_model_streamingly.py \
#      -c Test_Streamingly_Dual_Cache_RC_APC_24h_Chunk[100,49]_8ConformerLayer\
#      > train_logs/Test_Streamingly_Dual_Cache_RC_APC_24h_Chunk[100,49]_8ConformerLayer.log &

# CUDA_VISIBLE_DEVICES=0 nohup python test_A_cache_model_streamingly.py \
#      -c Test_Streamingly_Cache_RC_24h_Chunk[100,9]_8ConformerLayer_2\
#      > train_logs/Test_Streamingly_Cache_RC_24h_Chunk[100,9]_8ConformerLayer_2.log &

CUDA_VISIBLE_DEVICES=2 nohup python test_A_cache_model_streamingly.py \
     -c Test_Streamingly_Cache_RC_24h_Chunk[100,9]_8ConformerLayer_2\
     > train_logs/Test_Streamingly_Cache_RC_24h_Chunk[100,4]_8ConformerLayer_3.log &

# export CUDA_VISIBLE_DEVICES=2
# python test_A_dual_cache_model_streamingly.py -c Test_Streamingly_Dual_Cache_RC_24h_Chunk[100,49]_8ConformerLayer

# export CUDA_VISIBLE_DEVICES=0
# python cal_time_offline_model_buffered_2.py -c Test_Time_RC_24h_Offline_8ConformerLayer_Buffered_2_P0.9F0.1

# export CUDA_VISIBLE_DEVICES=0
# python test_A_offline_model_buffered_streaming_inf_2.py -c Test_RC_24h_Offline_8ConformerLayer_Buffered_2_P0.9F0.1

# export CUDA_VISIBLE_DEVICES=0
# python cal_time_cache_model_streamingly.py -c Test_Time_Streamingly_Cache_RC_24h_Chunk[100,49]_8ConformerLayer_2

# python test_A_result.py -c Test_RC_24h_Offline_8ConformerLayer_Buffered_P6F2


