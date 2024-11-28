# 查看npy文件
import numpy as np

data = np.load('./aki/aki_test_data.npy', allow_pickle=True)
print(data[20])