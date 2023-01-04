# coding=utf-8
import sys
import time
import math
import pickle
import numpy as np
import matplotlib.pyplot as plt


# 论文之中原始的参数
epochs_original, lr_global_original = 16, np.array([5e-4] * 2 + [2e-4] * 3 + [1e-4] * 3 + [5e-5] * 4 + [1e-5] * 8)
# 修改过的参数设置，轮数与学习率
# 100   0.05
epochs, lr_global_list = epochs_original, lr_global_original * 100


def train(LeNet5, train_images, train_labels):
    # 0.9   256
    momentum = 0.9
    weight_decay = 0
    batch_size = 256

    # 开始训练，循环训练轮数
    cost_last, count = np.Inf, 0
    err_rate_list = []
    for epoch in range(0, epochs):
        print("----------------------------------- epo {} begin -----------------------------------".format(epoch + 1))

        # Stochastic Diagonal Levenberg-Marquardt method 来决定学习率
        # 500 -> 256
        batch_image, batch_label = random_mini_batches(train_images, train_labels, mini_batch_size=500, one_batch=True)
        LeNet5.Forward_Propagation(batch_image, batch_label, 'train')
        lr_global = lr_global_list[epoch]
        # 0.02
        LeNet5.SDLM(0.02, lr_global)

        # 调试输出
        print("Global learning rate:", lr_global)
        print("Learning rates in layers:", np.array([LeNet5.C1.lr, LeNet5.C3.lr, LeNet5.C5.lr, LeNet5.F6.lr]))
        print("Batch size:", batch_size)

        # 对于每个选定的batch进行训练
        ste = time.time()
        cost = 0
        mini_batches = random_mini_batches(train_images, train_labels, batch_size)
        for i in range(len(mini_batches)):
            batch_image, batch_label = mini_batches[i]

            loss = LeNet5.Forward_Propagation(batch_image, batch_label, 'train')
            cost += loss
            LeNet5.Back_Propagation(momentum, weight_decay)

            # print progress
            # 可以不重定向输出
            if i % (int(len(mini_batches) / 100)) == 0:
                sys.stdout.write("\033[F")  # CURSOR_UP_ONE
                sys.stdout.write("\033[K")  # ERASE_LINE
                print("progress:", int(100 * (i + 1) / len(mini_batches)), "%, ", "cost =", cost, end='\r')
        sys.stdout.write("\033[F")  # CURSOR_UP_ONE
        sys.stdout.write("\033[K")  # ERASE_LINE

        # Loss??
        print("Cost of epo", epoch + 1, ":", cost, "                                             ")

        error01_train, _ = LeNet5.Forward_Propagation(train_images, train_labels, 'test')
        err_rate_list.append(error01_train / 60000)
        # error01_test, _ = LeNet5.Forward_Propagation(test_images, test_labels, 'test')
        # err_rate_list.append([error01_train / 60000, error01_test / 10000])
        print("Error number sum of training set:", error01_train, "/", len(train_labels))
        # print("0/1 error of testing set: ", error01_test, "/", len(test_labels))
        print("Time used: ", time.time() - ste, "second")
        print(
            "----------------------------------- epo {} end -------------------------------------\n".format(epoch + 1))


        # 使用pkl格式存储模型，也就是存储模型中的各个参数数据
        # if(epoch == epochs - 1):
            # with open('model_data.pkl', 'wb') as output:
                # pickle.dump(LeNet5, output, pickle.HIGHEST_PROTOCOL)
        # else:
            # pass

    err_rate_list = np.array(err_rate_list).T

    # 可以保留图像展示，整张图片
    # This shows the error rate of training and testing data after each epoch
    # x = np.arange(epochs)
    # plt.xlabel('epochs')
    # plt.ylabel('error rate')
    # plt.plot(x, err_rate_list[0])
    # plt.plot(x, err_rate_list[1])
    # plt.legend(['training data', 'testing data'], loc='upper right')
    # plt.show()


# 获取用于训练的batch数据
def random_mini_batches(image, label, mini_batch_size=256, one_batch=False):
    m = image.shape[0]  # number of training examples
    mini_batches = []

    # 获取随机生成的0-60000的序列
    permutation = list(np.random.permutation(m))
    shuffled_image = image[permutation, :, :, :]
    shuffled_label = label[permutation]

    # 仅仅获取一个batch进行训练
    if one_batch:
        mini_batch_image = shuffled_image[0: mini_batch_size, :, :, :]
        mini_batch_label = shuffled_label[0: mini_batch_size]
        return mini_batch_image, mini_batch_label

    # 对60000个数据进行数据分组
    # TODO
    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_image = shuffled_image[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_label = shuffled_label[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_image, mini_batch_label)
        mini_batches.append(mini_batch)
    # 处理最后一个数据分组，确定最后一个数据分组的大小
    if m % mini_batch_size != 0:
        mini_batch_image = shuffled_image[num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch_label = shuffled_label[num_complete_minibatches * mini_batch_size: m]
        mini_batch = (mini_batch_image, mini_batch_label)
        mini_batches.append(mini_batch)

    return mini_batches
