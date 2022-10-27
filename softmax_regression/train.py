# coding=utf-8
import numpy as np

from data_process import data_convert
from softmax_regression import softmax_regression


def train(train_images, train_labels, k, iters = 5, alpha = 0.5):
    m, n = train_images.shape # m = 60000, n = 28 * 28
    # data processing
    x, y = data_convert(train_images, train_labels, m, k) # x:[m,n], y:[10,m] x经过了灰度转换，y为标签矩阵，这里好像写错了(10)
    
    # Initialize theta.  Use a matrix where each column/row corresponds to a class,
    # and each row is a classifier coefficient for that class.
    theta = np.random.rand(k, n) # [k,n] k = 10, n = 28 * 28
    # do the softmax regression
    theta = softmax_regression(theta, x, y, iters, alpha) # x是特征矩阵，y是标签矩阵
    return theta

