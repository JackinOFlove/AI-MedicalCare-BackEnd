# 查看npy文件
import numpy as np

data = np.load('ards_test_data.npy', allow_pickle=True)
print(data[2])