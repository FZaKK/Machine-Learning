import random
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances

m = 19835
n = 624944

'''
temp = np.array([])
with open('sparse_triple_user.txt', 'r') as f:
    for line in f:
        line = line.strip('\n')
        user, label, flag = line.split()
        label = int(label)
        temp = np.append(temp, label)

print(int(temp.max()))
'''

'''
row_attribute = []
col_attribute = []
data_attribute = []
with open('sparse_triple_user.txt', 'r') as f:
    for line in f:
        line = line.strip('\n')
        cur_row, cur_col, cur_data = line.split()
        cur_row = int(cur_row)
        cur_col = int(cur_col)
        cur_data = int(cur_data)
        row_attribute.append(cur_row)
        col_attribute.append(cur_col)
        data_attribute.append(cur_data)
row_attribute = np.array(row_attribute)
col_attribute = np.array(col_attribute)
data_attribute = np.array(data_attribute)
# print(_row)
sparse_attribute_matrix = csr_matrix((data_attribute, (row_attribute, col_attribute)), shape=(m, n),
                                dtype=np.int32)  # 因为打分区间为100分，为了减小存储空间使用的是int类型
similarities_base_attribute = cosine_similarity(sparse_attribute_matrix)
# similarities = pairwise_distances(sparse_data_matrix, metric='jaccard')
print('pairwise dense output:\n {}\n'.format(similarities_base_attribute))
print(similarities_base_attribute.shape)
'''
number_of_users = 19835
label_numbers = 700000
row_attribute = [] # 用于处理属性扩展部分的稀疏矩阵
col_attribute = []
data_attribute = []
with open('sparse_triple_user.txt', 'r') as f:
    for line in f:
        line = line.strip('\n')
        cur_row, cur_col, cur_data = line.split()
        cur_row = int(cur_row)
        cur_col = int(cur_col)
        cur_data = int(cur_data)
        row_attribute.append(cur_row)
        col_attribute.append(cur_col)
        data_attribute.append(cur_data)
row_attribute = np.array(row_attribute)
col_attribute = np.array(col_attribute)
data_attribute = np.array(data_attribute)
# print(_row)
sparse_attribute_matrix = csr_matrix((data_attribute, (row_attribute, col_attribute)),
                                     shape=(number_of_users, label_numbers),
                                     dtype=np.int32)
print(sparse_attribute_matrix.__getitem__((0, 89968)))