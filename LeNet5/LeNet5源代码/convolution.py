import numpy as np
from data_preprocessing import zero_pad


# Numpy第三方库: np.tensordot()的快速优化
def conv_forward(A_prev, W, b, hyper_parameters):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape

    stride = hyper_parameters["stride"]
    pad = hyper_parameters["pad"]

    n_H = int((n_H_prev + 2 * pad - f) / stride + 1)
    n_W = int((n_W_prev + 2 * pad - f) / stride + 1)

    # Initialize the output volume Z with zeros.
    Z = np.zeros((m, n_H, n_W, n_C))
    A_prev_pad = zero_pad(A_prev, pad)
    for h in range(n_H):
        for w in range(n_W):
            # Use the corners to define the (3D) slice of a_prev_pad.
            A_slice_prev = A_prev_pad[:, h * stride:h * stride + f, w * stride:w * stride + f, :]
            # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron.
            Z[:, h, w, :] = np.tensordot(A_slice_prev, W, axes=([1, 2, 3], [0, 1, 2])) + b

    assert (Z.shape == (m, n_H, n_W, n_C))
    cache = (A_prev, W, b, hyper_parameters)
    return Z, cache


# 快速计算使用np.dot来加快计算
def conv_backward(dZ, cache):
    (A_prev, W, b, hyper_parameters) = cache
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    (m, n_H, n_W, n_C) = dZ.shape
    stride = hyper_parameters["stride"]
    pad = hyper_parameters["pad"]

    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))

    if pad != 0:
        A_prev_pad = zero_pad(A_prev, pad)
        dA_prev_pad = zero_pad(dA_prev, pad)
    else:
        A_prev_pad = A_prev
        dA_prev_pad = dA_prev

    for h in range(n_H):
        for w in range(n_W):
            # Find the corners of the current "slice"
            vert_start, horiz_start = h * stride, w * stride
            vert_end, horiz_end = vert_start + f, horiz_start + f

            # Use the corners to define the slice from a_prev_pad
            A_slice = A_prev_pad[:, vert_start:vert_end, horiz_start:horiz_end, :]

            # Update gradients for the window and the filter's parameters
            dA_prev_pad[:, vert_start:vert_end, horiz_start:horiz_end, :] += np.transpose(np.dot(W, dZ[:, h, w, :].T), (3, 0, 1, 2))

            dW += np.dot(np.transpose(A_slice, (1, 2, 3, 0)), dZ[:, h, w, :])
            db += np.sum(dZ[:, h, w, :], axis=0)

    # Set dA_prev to the unpadded dA_prev_pad
    dA_prev = dA_prev_pad if pad == 0 else dA_prev_pad[:, pad:-pad, pad:-pad, :]

    # Making sure your output shape is correct
    assert (dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))

    return dA_prev, dW, db


def conv_SDLM(dZ, cache):
    (A_prev, W, b, hparameters) = cache
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    (m, n_H, n_W, n_C) = dZ.shape
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))

    if pad != 0:
        A_prev_pad = zero_pad(A_prev, pad)
        dA_prev_pad = zero_pad(dA_prev, pad)
    else:
        A_prev_pad = A_prev
        dA_prev_pad = dA_prev

    for h in range(n_H):
        for w in range(n_W):
            # Find the corners of the current "slice"
            vert_start, horiz_start = h * stride, w * stride
            vert_end, horiz_end = vert_start + f, horiz_start + f

            # Use the corners to define the slice from a_prev_pad
            A_slice = A_prev_pad[:, vert_start:vert_end, horiz_start:horiz_end, :]

            # Update gradients for the window and the filter's parameters
            dA_prev_pad[:, vert_start:vert_end, horiz_start:horiz_end, :] += np.transpose(
                np.dot(np.power(W, 2), dZ[:, h, w, :].T), (3, 0, 1, 2))

            dW += np.dot(np.transpose(np.power(A_slice, 2), (1, 2, 3, 0)), dZ[:, h, w, :])
    # Set dA_prev to the unpaded dA_prev_pad
    dA_prev = dA_prev_pad if pad == 0 else dA_prev_pad[:, pad:-pad, pad:-pad, :]
    assert (dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
    return dA_prev, dW
