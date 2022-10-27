import os
import pandas as pd
import numpy as np
import math
import random
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing

K = 10  # 取前十个最相似的用户
user_list = []  # 用户列表用于之前的处理
number_of_items = 507172
number_of_users = 19835  # 经过对train.txt的统计得到的user的数量
num_of_cols = 624961
num_score = 5001507
sum_of_all_score = 247597504
label_numbers = 700000 # 624944 # 统计之中虽然只有7万个标签，但是最大值id为624943
basic_miu = sum_of_all_score / num_score
user_score_num = []  # 用户的打分的个数
_row = []
_col = []
_data = []
most_similar_users = {}
user_bias = {}

row_attribute = [] # 用于处理属性扩展部分的稀疏矩阵
col_attribute = []
data_attribute = []


class Sparse_triple:
    s = 0

    def __init__(self, u, i, s):
        self.user = u
        self.item = i
        self.score = s


def pearson_similarity(cal1, cal2):
    '''
    :param cal1:输入的第一个字符串列表
    :param cal2:输入的第二个字符串列表用于计算pearson相关系数
    :return:返回一个float数值,中间需要减去平均值
    '''
    cal1_mean = 0.0
    cal2_mean = 0.0
    sum1 = 0.0
    sum2 = 0.0
    for cal1_item in cal1:
        sum1 = sum1 + cal1_item.score
    for cal2_item in cal2:
        sum2 = sum2 + cal2_item.score
    cal1_mean = sum1 / len(cal1)
    cal2_mean = sum2 / len(cal2)
    dis1 = 0.0
    dis2 = 0.0
    for cal1_item in cal1:
        dis1 = dis1 + (cal1_item.score - cal1_mean) * (cal1_item.score - cal1_mean)
    for cal2_item in cal2:
        dis2 = dis2 + (cal2_item.score - cal2_mean) * (cal2_item.score - cal2_mean)
    dis1 = math.sqrt(dis1)
    dis2 = math.sqrt(dis2)
    ans = 0.0
    for triple1 in cal1:
        for triple2 in cal2:
            if triple1.item == triple2.item:
                ans = ans + (triple1.score - cal1_mean) * (triple2.score - cal2_mean)
            else:
                pass
    # print(dis1 * dis2)
    if dis1 * dis2 == 0:
        return 0.0
    else:
        ans = ans / (dis1 * dis2)
        return ans

# 如果有item属性和user属性相似就会提高预测的评分值
def item_expend(ans, user, item_to_pre, user_label_matrix):
    with open('itemAttribute.txt', 'r') as f:
        for line in f:
            line = line.strip('\n')
            item, attribute1, attribute2 = line.split('|')
            item = int(item)
            if item == item_to_pre:
                if attribute1 == 'None':
                    pass
                else:
                    attribute1 = int(attribute1)
                    if user_label_matrix.__getitem__((user, attribute1)) == 1:
                        ans = ans * (1 + 0.1)
                if attribute2 == 'None':
                    pass
                else:
                    attribute2 = int(attribute2)
                    if user_label_matrix.__getitem__((user, attribute2)) == 1:
                        ans = ans * (1 + 0.1)
            else:
                pass
    return ans


def pre_operate(temp_str):
    temp_str = temp_str.strip('\n')
    return temp_str


def list_append(line, cur_user1, cur_list):
    user_id, item_id, score = line.split()
    score = int(score)
    if cur_user1 == int(user_id):
        temp_triple = Sparse_triple(user_id, item_id, score)
        cur_list.append(temp_triple)
    return 0


def restruct_str(s):
    need_str = s
    need_str = need_str.strip('[')
    need_str = need_str.strip(']')
    return need_str


def predict(user_id, item_to_predict, user_similarity_matrix, data_matrix, label_matirx):
    '''
    :param user_id: 目前预测的user
    :param item_to_predict: 需要取预测该user对输入item的打分
    :param user_similarity_matrix: user相似度字典（前十）
    :param data_matrix: 数据评分稀疏矩阵
    :return:
    '''
    ans = 0
    sigema_up = 0
    sigema_down = 0
    data_array = np.array([])
    weight_array = np.array([])
    user_id = int(user_id)
    item_to_predict = int(item_to_predict)
    cur_sim_list = list(most_similar_users[user_id])  # 相似度前十的列表
    for similar_user in cur_sim_list:
        weight_array = np.append(weight_array, user_similarity_matrix[user_id][similar_user])  # 存储相关系数
    # print(weight_array)
    for similar_user in cur_sim_list:
        data_array = np.append(data_array, data_matrix.__getitem__((similar_user, item_to_predict)) + user_bias[similar_user])  # 存储评分+偏差
    # print(data_array)
    if np.all(weight_array == 0): # 考虑到皮尔森相关系数均为0的情形，即一个user评分都是一个值
        return round(user_bias[similar_user], 1)
    else:
        ans = np.average(data_array, weights=weight_array) * 0.8 + user_bias[user_id] * 0.2 # random.uniform(user_bias[user_id] - 10, user_bias[user_id] + 10) * 0.2
        ans = item_expend(ans, user_id, item_to_predict, label_matirx)
        return round(ans, 1)


''' 
with open('train.txt', 'r') as f: # 统计数据集相关内容，user数量，item数量等
    for line in f:
        if '|' in line:
            # number_of_users = number_of_users + 1
            str_list = line.split('|')
            user_list.append(str_list[0])
        
with open('itemAttribute.txt', 'r') as f:
    for line in f:
        str_list = line.split('|')
        with open('items.txt', 'a') as file:
            temp_str = str_list[0] + '\n'
            file.write(temp_str)
'''

'''  这里对数据中进行分析，发现每个user对至少10个item进行了打分
temp = {}
with open('train.txt', 'r') as f:
    for line in f:
        if '|' in line:
            line = line.strip('\n')
            str_list = line.split('|')
            temp[str_list[0]] = int(str_list[1])
print(min(temp.values()))
'''

'''
with open('train.txt', 'r') as f: # 处理得到三元组稀疏矩阵，并且已经将其减去了mean均值，得到了sparse_ready的存储
    for line in f:
        line = line.strip('\n')
        if '|' in line:
            cur_user_id, cur_score_num = line.split('|')
            cur_score_num = int(cur_score_num)
            user_score_num.append(cur_score_num)
#print(user_score_num)

with open('sparse_triple.txt', 'r') as f:
    for line in f:
        line = line.strip('\n')
        cur_row, cur_col, cur_data = line.split()
        cur_row = int(cur_row)
        cur_col = int(cur_col)
        cur_data = float(cur_data)
        _row.append(cur_row)
        _col.append(cur_col)
        _data.append(cur_data)
_row = np.array(_row)
_col = np.array(_col)
_data = np.array(_data)
# print(_row)
sparse_data_matrix = csr_matrix((_data, (_row, _col)), shape=(r, c), dtype=np.int32)
# sparse_data_matrix = preprocessing.scale(sparse_data_matrix, with_mean=False)
# similarities = cosine_similarity(sparse_data_matrix)
# print('pairwise dense output:\n {}\n'.format(similarities))
# print(sparse_data_matrix.toarray())
my_test = np.sum(sparse_data_matrix, axis=1)
for i in range(r):
    my_test[i][0] = my_test[i][0] / user_score_num[i]
    # print(int(my_test[i][0]))


user_id = ''
temp_dic = {}
with open('train.txt', 'r') as f:
    for line in f:
        line = line.strip('\n')
        if '|' in line:
            user, rate_num = line.split('|')
            user_id = user
            user_id = int(user_id)
        else:
            item_id, item_score = line.split()
            item_score = int(item_score)
            temp_str = str(user_id) + '  ' + item_id + '  ' + str(item_score - int(my_test[user_id][0])) + '\n'
            with open('user_bias.txt', 'a') as file:
                file.write(temp_str)
'''

'''
# 预先求好各个用户之间的相似度，两重循环，只取想要计算的两个用户的打分数据即可，完全自主实现的pearson相关系数计算
for i in range(6):
    list1 = []
    user_similarity = {}
    with open('temp_test.txt', 'r') as f:
        for line in f:
            user_id, item_id, score = line.split()
            score = int(score)
            if i == int(user_id):
                temp_triple = Sparse_triple(user_id, item_id, score)
                list1.append(temp_triple)
    # for ans in list1:
        # print(ans.user + ' ' + ans.item + ' ' + str(ans.score))

    for j in range(6):
        list2 = []
        with open('temp_test.txt', 'r') as f:
            for line in f:
                user_id, item_id, score = line.split()
                score = int(score)
                if j == int(user_id):
                    temp_triple = Sparse_triple(user_id, item_id, score)
                    list2.append(temp_triple)
        if i != j:
            user_similarity[j] = pearson_similarity(list1, list2)
    user_similarity = sorted(user_similarity.items(), key=lambda x: x[1], reverse=True)
    user_similarity = list(user_similarity)
    with open('temp_similarity.txt', 'a') as file:
        for k in range(2):
            file.write(str(i) + ' ' + str(user_similarity[k][0]) + ' ' + str(user_similarity[k][1]) + '\n')
'''

'''
k = 0 # 这里是用于取train训练集中的每一个用户的前三个打分项作为测试集，检验RMSE
with open('train.txt', 'r') as f:
    for line in f:
        line = line.strip('\n')
        if '|' in line:
            k = 0
            user_id, predict_num = line.split('|')
            with open('RMSE_test.txt', 'a') as file:
                file.write(user_id + '|' + '3' + '\n')
            print(user_id)
        elif k < 3:
            item_id, cur_score = line.split()
            print(item_id)
            with open('RMSE_test.txt', 'a') as file:
                file.write(item_id + '\n')
            k = k + 1
'''

''' 
temp = np.array([])
with open('sparse_triple_user.txt', 'r') as f: # 利用物品的标签绘制用户画像，统计label的特征
    for line in f:
        line = line.strip('\n')
        user, label, flag = line.split()
        label = int(label)
        temp = np.append(temp, label)

print(int(temp.max()))
'''

if __name__ == '__main__':
    with open('users.txt', 'r') as f:
        for line in f:
            line = line.strip('\n')
            user, bias = line.split()
            user = int(user)
            bias = float(bias)
            user_bias[user] = bias

    with open('sparse_ready.txt', 'r') as f: # 这里可以考虑考虑
        for line in f:
            line = line.strip('\n')
            cur_row, cur_col, cur_data = line.split()
            cur_row = int(cur_row)
            cur_col = int(cur_col)
            cur_data = float(cur_data)
            _row.append(cur_row)
            _col.append(cur_col)
            _data.append(cur_data)
    _row = np.array(_row)
    _col = np.array(_col)
    _data = np.array(_data)
    # print(_row)
    sparse_data_matrix = csr_matrix((_data, (_row, _col)), shape=(number_of_users, num_of_cols),
                                    dtype=np.int32)  # 因为打分区间为100分，为了减小存储空间使用的是int类型
    similarities = cosine_similarity(sparse_data_matrix)  # 已经转换原来的值求得皮尔森相关系数

    # 这里是基于label的稀疏矩阵相似度处理
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
    sparse_attribute_matrix = csr_matrix((data_attribute, (row_attribute, col_attribute)), shape=(number_of_users, label_numbers),
                                         dtype=np.int32)
    # similarities = pairwise_distances(sparse_data_matrix, metric='jaccard')


    for i in range(number_of_users):
        for j in range(number_of_users):
            if i == j:
                similarities[i][j] = -99  # 消除1的影响，仅让similarities变换即可

    for i in range(number_of_users):
        most_similar_users[i] = list(np.argpartition(similarities[i], -K)[-K:])
        # print(i, np.argpartition(similarities[i], -K)[-K - 1:][:-1]) # 要去除pearson相关系数为1的情形
    print('pairwise dense output:\n {}\n'.format(similarities))
    # print(most_similar_users)
    # print(sparse_data_matrix)
    # print(sparse_data_matrix.__getitem__((0,0)), 'success')

    user_id = ''
    predict_num = ''
    with open('result.txt', 'a') as f:
        f.truncate(0)
    with open('test.txt', 'r') as f:
        for line in f:
            if '|' in line:
                with open('result.txt', 'a') as file:
                    file.write(line)
                user_id, predict_num = line.split('|')
                predict_num = int(predict_num)
                # print(user_id)
            else:
                predict_item = line.strip('\n')
                pre_score = predict(user_id, predict_item, similarities, sparse_data_matrix, sparse_attribute_matrix)
                print(pre_score)
                with open('result.txt', 'a') as file:
                    file.write(predict_item + '  ' + str(pre_score) + '\n')
