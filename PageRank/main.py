import networkx as nx
import numpy as np
import linecache

total_num = 0  # 预先统计得到的节点数目，用于（1-β）/N，本数据集结点数为6263
alpha = 0.85  # 参数
e = 0.0001  # 阈值
diff = 1  # 判断迭代是否继续的差值,1是为了使第一次迭代开始
total_end_list = []  # 所包含的所有的节点[1, 2...],因为6263可以存储就简略了读取文本部分
dead_end = []  # 黑洞结点的处理，直接在迭代过程中补上即可

# 这三个变量用于输出结果
node_list = []
score_list = []
node_dict = {}


# 用于统计所有的节点
def count():
    with open('data.txt', 'r') as f:
        for line in f:
            src, destination = line.split()
            src = int(src)
            destination = int(destination)
            if src not in total_end_list:
                total_end_list.append(src)
            else:
                pass
            if destination not in total_end_list:
                total_end_list.append(destination)
            else:
                pass
    total_end_list.sort()
    return len(total_end_list)


# 创建稀疏矩阵的链表表示形式，存入src-des.txt
def sparse_list():
    for i in range(total_num):
        degree = 0
        rival_end = []
        with open('data.txt', 'r') as f:
            for line in f:
                src, destination = line.split()
                src = int(src)
                destination = int(destination)
                if src == total_end_list[i]:
                    degree = degree + 1
                    rival_end.append(destination)
                else:
                    pass
        with open('src-des.txt', 'a') as f:
            f.write(str(src) + '-' + str(degree) + '-' + str(rival_end) + '\n')


# 将所得的形如'[]'的字符串转化为数字字符串组成的列表
def str2list(str):
    str_list = []
    str = str.strip('[')
    str = str.strip('\n')
    str = str.strip(']')
    str_list = str.split(', ')
    return str_list


# 重构字符串
def restruct_str(str):
    str = str.strip('[')
    str = str.strip('\n')
    str = str.strip(']')
    return str


# 实现block-stripe算法初始化评价值
def init():
    with open("result.txt", 'r+') as f:  # 用来每次输出结果前先把文本清空
        f.truncate(0)
    with open('result.txt', 'r+') as f:
        for i in range(total_num):
            f.write(str(total_end_list[i]) + ' ' + str(1 / total_num))


# 在存每一个节点的list的时候要进行排序，注意dead_end的处理，把dead_end先存到一个list里
# 主函数：运行调试
if __name__ == "__main__":
    '''
    with open('src-des.txt', 'r') as f:
        for line in f:
            src, degree, des = line.split('-')
            print(len(str2list(des)))
    '''
    '''
    temp_dict={}
    with open('src-des.txt', 'r') as f:  # 这里得处理dead-end!!!!
        for line in f:
            src, degree, destination = line.split('-')
            degree = int(degree)
            if src == '63':
                print(degree)
                print(str2list(destination))
            if '6' in str2list(destination) :  # 这里不能简单的进行判断，有bug
                temp_dict[src] = degree
    print(temp_dict)
    '''

    with open('data.txt', 'r') as f:
        for line in f:
            src, destination = line.split()
            src = int(src)
            destination = int(destination)
            if src not in total_end_list:
                total_end_list.append(src)
            else:
                pass
            if destination not in total_end_list:
                total_end_list.append(destination)
            else:
                pass
    total_end_list.sort()
    total_num = len(total_end_list)
    # print(total_end_list)

    with open("temp.txt", 'r+') as f:  # 用来每次输出结果前先把文本清空,初始化
        f.truncate(0)
    with open('temp.txt', 'r+') as f:
        for i in range(total_num):
            f.write('[{}] [{}]\n'.format(total_end_list[i], 1 / total_num))

    # 将dead-end存入list中
    with open('src-des.txt', 'r') as f:
        for line in f:
            src, degree, destination = line.split('-')
            degree = int(degree)
            if degree == 0:
                dead_end.append(src)
    print(dead_end)  # 输出 正确
    print(len(dead_end))

    # 这里如果因为结点数过多，需要再次添加一个txt文本，来更新每次迭代的评价值
    # 因为数据集只有6263个结点，就存储在内存
    # for k in range(3):
    while diff > e:  # 判断阈值或者阈值为0
        cur_node_list = []
        new_score_list = []
        diff = 0
        for i in range(1, total_num + 1):
            linecache.updatecache('temp.txt')
            cur_line = linecache.getline('temp.txt', i).strip('\n')  # 取出每一行要处理的结点信息,哪些结点指向了该结点
            cur_node, old_score = cur_line.split()
            cur_node = cur_node.strip('[')
            cur_node = cur_node.strip(']')
            old_score = old_score.strip('[')
            old_score = old_score.strip(']')
            my_dict = {}  # 包含入度结点的信息
            new_score = 0.0
            # print(float(old_score))
            with open('src-des.txt', 'r') as f:  # 这里得处理dead-end!!!!
                for line in f:
                    src, degree, destination = line.split('-')
                    degree = int(degree)
                    if cur_node in str2list(destination):  # 这里不能简单的进行判断，有bug
                        my_dict[src] = degree
            # print(my_dict)
            # print(len(my_dict))
            for item in dead_end:  # 相当于初始化矩阵
                my_dict[item] = total_num
            # print(my_dict)
            # print(len(my_dict))
            f = open('temp.txt', 'r+')
            for temp_line in f:
                temp_node, temp_score = temp_line.split()  # 遍历原来对应矩阵一行的数据来获得新的评价值
                temp_node = temp_node.strip('[')
                temp_node = temp_node.strip(']')
                temp_score = temp_score.strip('[')
                temp_score = temp_score.strip(']')
                # print(temp_node + ' ' + temp_score)
                if temp_node in list(my_dict.keys()):
                    try:
                        new_score = new_score + float(temp_score) * 1 / my_dict[temp_node]  # 有三个相同的边都是75->8，结点75的出度24个结点
                    except:
                        pass
            new_score = new_score * 0.85 + 0.15 * 1 / total_num  # 要么乘以列表中的系数，要么将出度degree降低，删除列表中的重复元素
            # print(diff)
            if (abs(new_score - float(old_score))) > diff:  # 计算每次迭代的最大的差值判断迭代是否继续,这里的temp_score有问题
                diff = abs(new_score - float(old_score))
            else:
                pass
            cur_node_list.append(cur_node)  # 用于更新文本数据
            new_score_list.append(new_score)
            # print(my_dict)
            # print(diff)
            # print(new_score)  # 调试结点新的评价值
            f.close()
        with open('temp.txt', 'r+') as f:  # 用来每次迭代更新输出结果前先把文本清空
            f.truncate(0)
        with open('temp.txt', 'a') as f:
            for j in range(total_num):
                f.write('[{}] [{}]\n'.format(cur_node_list[j], new_score_list[j]))

    with open('temp.txt', 'r') as f:
        for line in f:
            node, score = line.split()
            node = restruct_str(node)
            score = restruct_str(score)
            node_list.append(node)
            score_list.append(score)
    for i in range(len(node_list)):
        node_dict[node_list[i]] = float(score_list[i])
    node_dict = sorted(node_dict.items(), key=lambda x: x[1], reverse=True)

    with open("result.txt", 'r+') as f:  # 用来每次输出结果前先把文本清空
        f.truncate(0)
    with open('result.txt', 'a') as f: # encoding='UTF-16 LE'
        for i in range(100):
            f.write('[{}] [{}]\n'.format(node_dict[i][0], node_dict[i][1]))

    '''
    # 初始化评价值 用于在result.txt中生成最后的结果
    with open("result.txt", 'r+') as f:  # 用来每次输出结果前先把文本清空
        f.truncate(0)
    with open('result.txt', 'r+') as f:
        for i in range(total_num):
            f.write('[{}] [{}]\n'.format(total_end_list[i], 1 / total_num))
    '''

    '''
    # 这一部分使针对y，a，m上课讲的例子实现的block-stripe，属于3个结点的小数据
    temp_list = ['y', 'a', 'm']
    with open("temp.txt", 'r+') as f:  # 用来每次输出结果前先把文本清空
        f.truncate(0)
    with open('temp.txt', 'r+') as f:
        for i in range(3):
            f.write('[{}] [{}]\n'.format(temp_list[i], 1 / 3))

    # 将dead-end存入list中
    with open('testlist.txt', 'r') as f:
        for line in f:
            src, degree, destination = line.split('-')
            degree = int(degree)
            if degree == 0:
                dead_end.append(src)
    print(dead_end) #输出

    # 这里如果因为结点数过多，需要再次添加一个txt文本，来更新每次迭代的评价值
    # 因为数据集只有6263个结点，就存储在内存
    #for k in range(2):
    while diff>e: #判断阈值或者阈值为0
        cur_node_list = []
        new_score_list = []
        diff = 0
        for i in range(1, 4):
            linecache.updatecache('temp.txt')
            cur_line = linecache.getline('temp.txt', i).strip('\n')  # 取出每一行要处理的结点信息,哪些结点指向了该结点
            print(cur_line)
            cur_node, old_score = cur_line.split()
            cur_node = cur_node.strip('[')
            cur_node = cur_node.strip(']')
            old_score = old_score.strip('[')
            old_score = old_score.strip(']')
            my_dict = {}  # 包含入结点的信息
            new_score = 0.0
            #print(float(old_score))
            with open('testlist.txt', 'r') as f:  # 这里得处理dead-end!!!!
                for line in f:
                    src, degree, destination = line.split('-')
                    degree = int(degree)
                    if cur_node in destination:
                        my_dict[src] = degree
            for item in dead_end:  # 相当于初始化矩阵
                my_dict[item] = 3
            print(my_dict)
            f = open('temp.txt', 'r+')
            for temp_line in f:
                temp_node, temp_score = temp_line.split()  # 遍历原来对应矩阵一行的数据来获得新的评价值
                temp_node = temp_node.strip('[')
                temp_node = temp_node.strip(']')
                temp_score = temp_score.strip('[')
                temp_score = temp_score.strip(']')
                print(temp_node + ' ' + temp_score)
                if temp_node in list(my_dict.keys()):
                    new_score = new_score + float(temp_score) * 1 / my_dict[temp_node]
            new_score = new_score * 0.80 + 0.20 * 1 / 3
            #print(diff)
            if (abs(new_score - float(old_score))) > diff:  # 计算每次迭代的最大的差值判断迭代是否继续,这里的temp_score有问题
                diff = abs(new_score - float(old_score))
            else:
                pass
            cur_node_list.append(cur_node)  # 用于更新文本数据
            new_score_list.append(new_score)
            #print(my_dict)
            print(diff)
            print(new_score)
            f.close()
        with open('temp.txt', 'r+') as f:  # 用来每次迭代更新输出结果前先把文本清空
            f.truncate(0)
        with open('temp.txt', 'a') as f:
            for j in range(3):
                f.write('[{}] [{}]\n'.format(cur_node_list[j], new_score_list[j]))
    '''

    '''
    :已经成功运行得到了稀疏矩阵的链表表示，目前这里也没啥用
    finished_end_list = []
    for i in range(total_num):
        degree = 0  # 结点的出度
        cur_destination = []  # 结点指向的结点
        if total_end_list[i] not in finished_end_list:
            with open('data.txt', 'r') as f:
                for line in f:
                    src, destination = line.split()
                    src = int(src)
                    destination = int(destination)
                    if src == total_end_list[i] and destination not in cur_destination:
                        degree = degree + 1
                        cur_destination.append(destination)
                    else:
                        pass
            # f degree != 0:
            with open('src-des.txt', 'a') as f:
                f.write(str(total_end_list[i]) + '-' + str(degree) + '-' + str(cur_destination) + '\n')
    '''

'''
import networkx as nx #使用networkx库的方法

# 创建有向图
G = nx.DiGraph()

#读取data数据
f = open(r"data.txt", "r")
for line in f:
    src, destination = line.split()
    src = int(src)
    destination = int(destination)
    G.add_edge(src, destination)
f.close()

# 有向图之间边的关系
pagerank_list = nx.pagerank(G, alpha=0.85)
# print("pagerank 值是：", pagerank_list)
pagerank_list = dict(sorted(pagerank_list.items(), key=lambda x: x[1], reverse=True))  # 对字典进行排序
key = list(pagerank_list.keys())  # 分成两个列表以方便输出
value = list(pagerank_list.values())

#输出结果保持了编码格式的相同均为UTF-8
with open(r"result.txt", 'r+') as file:  # 用来每次输出结果前先把文本清空
    file.truncate(0)
for i in range(len(key)):  # 将字典数据写入文本
    with open(r'result.txt', 'a') as f:
        # f.write(key[i] + " : " + str(value[i]))
        f.write('{} : {}\n'.format(key[i], value[i]))
'''
