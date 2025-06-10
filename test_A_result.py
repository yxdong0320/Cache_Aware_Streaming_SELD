from utils.cls_tools.cls_compute_seld_results import ComputeSELDResults
import argparse
import os
import yaml
dcase_output_val_dir = \
    '/disk6/yxdong/cache_aware_streaming_SELD/results/Test_Streamingly_Cache_RC_24h_Chunk[100,4]_8ConformerLayer_2/results'
score_obj = \
      ComputeSELDResults(ref_files_folder='/disk6/yxdong/cache_aware_streaming_SELD/dataset/metadata_dev_test')
val_ER, val_F, val_LE, val_LR, val_seld_scr, classwise_test_scr = score_obj.get_SELD_Results(dcase_output_val_dir)
print('ER/F/LE/LR/SELD: {}'.format('{:0.4f}/{:0.4f}/{:0.4f}/{:0.4f}/{:0.4f}'.format(val_ER, val_F, val_LE, val_LR, val_seld_scr)))
print('Class\tER\tF\tLE\tLR\tSELD_score')
for cls_cnt in range(0,13):
    print('{}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}'.format(
        cls_cnt,
        classwise_test_scr[0][cls_cnt],
        classwise_test_scr[1][cls_cnt],
        classwise_test_scr[2][cls_cnt],
        classwise_test_scr[3][cls_cnt],
        classwise_test_scr[4][cls_cnt]))