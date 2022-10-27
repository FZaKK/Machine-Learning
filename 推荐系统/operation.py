import numpy as np
import math
# from sklearn.metrics import mean_squared_error # 均方误差

org = np.array([])
pre = np.array([])
with open('RMSE_origin.txt', 'r') as f:
    for line in f:
        line = line.strip('\n')
        if '|' in line:
            pass
        else:
            item, score = line.split()
            score = float(score)
            org = np.append(org, score)
with open('RMSE_result.txt', 'r') as f:
    for line in f:
        line = line.strip('\n')
        if '|' in line:
            pass
        else:
            item, score = line.split()
            score = float(score)
            pre = np.append(pre, score)

# print(org[:100])
# print(pre[:100])
RMSE = np.sqrt(np.mean(np.square(pre - org)))
print(RMSE)
