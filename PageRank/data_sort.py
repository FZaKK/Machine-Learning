node_list = []
score_list = []
node_dict = {}

# 重构字符串
def restruct_str(str):
    str = str.strip('[')
    str = str.strip('\n')
    str = str.strip(']')
    return str

with open('temp.txt', 'r+') as f:
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
with open('result.txt', 'a', encoding='UTF-16 LE') as f:
    for i in range(100):
        f.write('[{}] [{}]\n'.format(node_dict[i][0], node_dict[i][1]))
