# coding=utf-8
import numpy as np


def predict(test_images, theta):
    scores = np.dot(test_images, theta.T) # test_images: [60000, 28*28] theta: [10, 28*28]
    preds = np.argmax(scores, axis=1) # 矩阵乘法乘出来十个class的不同概率选最大的
    return preds

# y_pred: [10000, 1]  y in main(test_labels): [10000, 1] 直接计算
def cal_accuracy(y_pred, y):
    # TODO: Compute the accuracy among the test set and store it in acc
    # for循环可以优化

    # 预测值与原标签相同则 +1
    score = 0
    for row in range(y.shape[0]):
        if(y_pred[row][0] == y[row][0]):
            score = score + 1

    # print(temp_y_pred)
    accuracy = score / y.shape[0]

    return accuracy