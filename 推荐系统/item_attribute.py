import linecache
import numpy as np
import math
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
k = 624960 #项目（item）总数
item = []
similarity_res = np.matrix
T = []#标签集合
number_all_item = 71880#标签总数
labels = [[]]#用户标签列表集合
labeled_users = []
number_of_attributes = 0
user_list = []
number_of_users = 19835
user_label_list = []

_row = []
_col = []
_data = []

class Sparse_triple:
    s = 0
    def __init__(self, u, i, s):
        self.user = u
        self.item = i
        self.score = s

def identifier_1(i, j, item):#计算不同标签是否共现
    if str(i) in item and str(j) in item:
        return 1
    else:
        return 0

def identifier_2(i, item):#计算单个标签是否出现
    if str(i) in item:
        return 1
    else:
        return 0
def cor(i,j): #计算标签相关度
    sum_1 = 0
    sum_2 = 0
    for x in range(number_of_users, 100):
        sum_1 = sum_1 + identifier_1(i, j, labeled_users[x])
    for y in range(number_of_users, 100):
        sum_2 = sum_2 + identifier_2(i, labeled_users[y])
    return sum_1/sum_2

def simt(item_i,item_j,k_mi,k_mj): #计算两个用户之间基于标签的相关度
    sum_3 = 0
    for i in range(k_mi):
        for j in range(k_mj):
            sum_3 = sum_3+cor(labeled_users[item_i][i], labeled_users[item_j][j])
    if k_mi*k_mj == 0:
        return 0
    return sum_3/(k_mi*k_mj)

def user_similarity():
    '''with open('labeled_user.txt', 'r') as f:#生成用户-标签三元组
        list_temp = []
        for line in f:
            list_temp = []
            list_temp.extend(line.split('|'))
            list_temp = list_temp[:-1]
            print(list_temp)
            if not len(list_temp) == 1:
                with open('sparse_triple_user.txt', 'a') as file:
                    for i in range(1, len(list_temp)):
                        str_temp = list_temp[0] + ' ' + list_temp[i] + ' ' + '1' + '\n'
                        file.write(str_temp)
    '''



def calculate_user_labels():
    '''with open('train.txt')as f:
       for line in f:
           if '|' in line:
                str_list = line.split('|')
                user_list.append(str_list[0])
    user_id = ''
    with open('train.txt', 'r') as f:
        for line in f:
            line = line.strip('\n')
            if '|' in line:
                user, rate_num = line.split('|')
                user_id = user
            else:
                item_id, item_score = line.split()
                #with open('itemAtrribute.txt') as file_0:
                str = user_id + '  ' + item_id + '  ' + item_score + '\n'
                with open('sparse_triple.txt', 'a') as file:
                    file.write(str)
'''
    for i in range(number_of_users):
        labels.append([])
        #labeled_users.append([])
    with open('sparse_triple.txt', 'r') as file:
        for line in file:
            user_id, item_id, item_score = line.split('  ')
            #print(user_id)
            if int(item_score) >= 90: #用户评价达到90分阈值则计入标签
                labels_str = linecache.getline('itemAttribute.txt', int(item_id))
                labels_str = labels_str[:-1]
                if labels_str!='':
                    #print(labels_str)
                    #print(labels_str.split('|'))
                    id, attribute_1, attribute_2 = labels_str.split('|')
                    labels[int(user_id)].append(attribute_1)
                    labels[int(user_id)].append(attribute_2)

    for user_label in labels:#给每个用户打上标签
        num = []
        for i in user_label:
            num.append(user_label.count(i))

        list_num = len(set(num))
        total_types = [[] * i for i in range(list_num)]
        times = [[] * i for i in range(list_num)]
        for i in range(list_num, 0, -1):
            for j in range(len(user_label)):
                if i == num[j]:
                    total_types[i - 1].append(user_label[j])
                    times[i - 1].append(num[j])

        total_types.reverse()
        times.reverse()
        new_list_type = [[] * i for i in range(len(times))]
        for i in range(len(times)):#标签去重
            types_set = set(total_types[i])
            length = len(types_set)
            new_list_type[i] = sorted(types_set, reverse=True)
            del [times[i][length:]]

        final_result = []
        for (type_line, times_line) in zip(new_list_type, times):
            for (type_element, times_element) in zip(type_line, times_line):
                #final_result.append("{0} {1}".format(type_element, times_element)) #这句统计包含了每个标签的出现次数，以后优化可能会用到
                final_result.append(type_element)
        final_result = final_result[:5]#截取前n个标签
        labeled_users.append(final_result)


#这部分写不写文档都行，内存能存下来
    '''with open('labeled_user.txt','a') as f:
        user_id = 0
        for user in labeled_users:
            str_1 = ''
            for i in user:
                if i != 'None':
                    str_1 = str_1 + i + '|'
            str_1 = str(user_id)+'|'+str_1+'\n'
            f.write(str_1)
            user_id = user_id + 1
'''
def get_labels_similarity():
    for i in range(0, number_of_users,195):
        for j in range(i, number_of_users, 195):
            if not i == j:

                sim = simt(i, j, len(labeled_users[i]), len(labeled_users[j]))
                #print(i, j, sim)
                #similarity_res[i][j] = sim
                #直接写三元组
                if not sim == 0:
                    with open('result_label_similarity.txt', 'a') as f:
                        f.write(str(i)+' '+str(j)+' '+str(sim)+"\n")

def get_all_labels():#获取到所有的标签集合
    with open ('itemAttribute.txt','r') as f:
        for line in f:
            id, attribute_1, attribute_2 = line.split('|')
            if not attribute_1 in T:
                T.append(attribute_1)
            if not attribute_2 in T:
                T.append(attribute_2)




if __name__ == '__main__':
    matrix_temp = np.zeros((number_of_users, number_all_item))
    #calculate_user_labels()
    print('PreOption Done.')
    '''similarity_res = [None]*number_of_users
    for i in range(number_of_users):
        similarity_res[i] = [0] * number_of_users
        '''
    with open('sparse_triple_user.txt', 'r') as f:
        for line in f:
            line = line.strip('\n')
            cur_row, cur_col, cur_data = line.split()
            matrix_temp[int(cur_row)][int(cur_col)] = cur_data

    # print(_row)
    '''sparse_data_matrix = csr_matrix((_data, (_row, _col)), shape=(number_of_users, number_all_item),
                                    dtype=np.int32)
    similarities = cosine_similarity(sparse_data_matrix)
    '''
    #user_similarity()
    #get_all_labels()
    #get_labels_similarity()
    print('Simt Done.')
    '''with open('result_label_similarity.txt', 'a') as f:
        for i in range(number_of_users):
            for j in range(number_of_users):
                str_1 = str(i)+' '+str(j) + ' ' + str(similarity_res[i][j])
                f.write(str_1)
'''