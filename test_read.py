import glob
import os
import pdb

from pathlib import Path


group2_path="/disk6/yxdong/cache_aware_streaming_SELD/results/Test_Streamingly_Cache_RC_24h_Chunk[100,4]_8ConformerLayer_3/results"
p = Path(group2_path)
csv_files = list(p.glob("*.csv"))
csv_files = [str(f) for f in csv_files]  # 转换为字符串路径
# csv_files = glob.glob(os.path.join(group2_path, "*.csv"))
# pdb.set_trace()
print(csv_files)