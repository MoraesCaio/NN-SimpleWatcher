import numpy as np


def padding(np_ar, kernel, extension=False):
    # adding margins
    hks = int((kernel.shape[0] - 1) / 2)  # half_kernel_size
    padded_ar = np.zeros([np_ar.shape[0] + 2 * hks, np_ar.shape[1] + 2 * hks, np_ar.shape[2]], np.uint8)
    padded_ar[hks:-hks, hks:-hks] = np.copy(np_ar)

    if not extension:
        # zero padding
        return padded_ar

    # extension padding on edges
    padded_ar[:hks, :] = padded_ar[hks:hks + 1, :]
    padded_ar[:, :hks] = padded_ar[:, hks:hks + 1]
    padded_ar[-hks:, :] = padded_ar[-(hks + 1):-hks, :]
    padded_ar[:, -hks:] = padded_ar[:, -(hks + 1):-hks]

    return padded_ar


def convolution(np_ar, kernel, stride=1, bias=0):
    ein_op1 = 'yx' if len(np_ar.shape) == 2 else 'yxc'
    ein_op2 = 'yx' if len(kernel.shape) == 2 else 'yxc'
    ein_eq = ein_op1 + ',' + ein_op2 + '->'

    dim = kernel.shape[0]
    hks = dim // 2  # half kernel size
    conv_shape0 = ((np_ar.shape[0] - dim) // stride) + 1
    conv_shape1 = ((np_ar.shape[1] - dim) // stride) + 1
    conv = np.zeros((conv_shape0, conv_shape1))

    j = 0
    for y in range(dim - 1, np_ar.shape[0], stride):
        i = 0
        for x in range(dim - 1, np_ar.shape[1], stride):
            window = np_ar[(y - dim + 1): (y + 1), (x - dim + 1): (x + 1)]
            conv[j - hks, i - hks] = np.einsum(ein_eq, window, kernel) + bias
            i += 1
        j += 1

    return conv


def conv_colors(np_ar, kernel_2d, stride=1, bias=0):

    crop_dim = kernel_2d.shape[0] - 1
    shape = (np_ar.shape[0] - crop_dim, np_ar.shape[1] - crop_dim, np_ar.shape[2])

    copy = np.zeros(shape, dtype=np_ar.dtype)

    for i in range(copy.shape[2]):
        copy[:, :, i] = convolution(np_ar[:, :, i], kernel_2d, stride, bias)

    return copy


def format_img_ar(ar):
    return np.uint8(np.clip(ar, 0, 255))


def relu(np_ar):
    return np.maximum(0, np_ar)


def tanh(np_ar):
    return (2 / (1 + np.exp(-2 * np_ar))) - 1


def np_ar_from_text():
    pass


def normalize():
    # mininum value -> 0
    # maximum value -> 255
    pass
