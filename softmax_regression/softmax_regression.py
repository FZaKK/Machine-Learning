# coding=utf-8
import numpy as np
np.set_printoptions(threshold=np.inf)


# softmax函数，输出一个一维数组
def softmax(s):
    return np.exp(s) / np.sum(np.exp(s), axis=1)

# m = 60000, n = 28 * 28
# theta: [10, 28*28]  x:[m,n], y:[10,m] x经过了灰度转换，y为标签矩阵  iters: 迭代次数  alpha: 学习率
# 主要就是求theta系数矩阵
def softmax_regression(theta, x, y, iters, alpha):
    # TODO: Do the softmax regression by computing the gradient and 
    # the objective function value of every iteration and update the theta
    # print(y.shape[0])
    # print(y.shape[1])
    for my_iter in range(iters):
        f = 0  # 每次迭代的损失值
        g = np.zeros((theta.shape[0], theta.shape[1])) # 迭加迭代的梯度下降
        for i in range(x.shape[0]):
            y_probal = softmax(np.dot(x[i].reshape(1, -1), theta.T)) # y_probal: [1, 10]
            f = f + np.dot(y.T[i].reshape(-1, 1), np.log(y_probal))
            error = (y.T[i] - y_probal).T
            # print(error)
            # print(error)
            # print(x[i].reshape(1, -1))
            delta_gredient = np.dot(error, x[i].reshape(1, -1))
            # print(delta_gredient.shape[0])
            # print(delta_gredient.shape[1])
            # print(delta_gredient[0])
            g = g + delta_gredient
        f = (- 1 / x.shape[0]) * f
        g = (- 1 / x.shape[0]) * g
        theta = theta - alpha * g  # 全部样本梯度下降算法公式
        # print('损失值: ', f)
        # print('梯度下降: ', g)
        print('k:', my_iter, 'weights:', theta[5][36])

    return theta
    
