from PIL import Image as img
import numpy as np
import deep_utils as du
from img_filter import Filter as F


def to_gray(np_ar):

    factor = np.ones(np_ar.shape[2], dtype='float64')
    factor[:3] = np.array([0.299, 0.587, 0.114])

    y = np_ar[:, :, :3].dot(factor[:3].T)

    return np.int64(y)


if __name__ == '__main__':

    path = 'ImagensDeTeste/'
    files = ['CNN1.png', 'IMG_0103.png', 'testpat.1k.png']

    filter_names = [
        'gx',
        'gy',
        'sharpen3',
        'mean3',
        'mean5',
        'gaussian3',
        'gaussian5',
        'unsharp5'
    ]

    filters = [
        F.kernels['gx'],
        F.kernels['gy'],
        F.kernels['sharpen3'],
        F.kernels_float['mean3'],
        F.kernels_float['mean5'],
        F.kernels_float['gaussian3'],
        F.kernels_float['gaussian5'],
        F.kernels_float['unsharp5']
    ]

    # 2D convolution
    for file in files:
        image_ar = np.int64(img.open(path + file))
        for n, f in zip(filter_names, filters):
            gray = to_gray(image_ar)
            result = du.convolution(du.padding(gray, f, extension=True), f)
            result = du.format_img_ar(result)
            img.fromarray(result).save(path + file[:-4] + '_' + n + '.jpg', mode='L')

    # 3D convolution

    # ff stands for first_filter
    ff = np.ones((3, 3, 3), dtype='float64')
    ff[:, :, 0] = F.kernels['gx']
    ff[:, :, 1] = F.kernels['gy']
    ff[:, :, 2] = F.kernels_float['mean3']

    for file in files:
        image_ar = np.int64(img.open(path + file))
        result = du.convolution(du.padding(image_ar, ff, extension=True), ff)
        result = du.format_img_ar(result)
        img.fromarray(result).save(path + file[:-4] + '_gx_gy_mean3.jpg')

    for file in files:
        image_ar = np.int64(img.open(path + file))
        for color in range(3):
            image_ar[:, :, color] = du.convolution(du.padding(image_ar[:, :, color], ff[:, :, color], extension=True), ff[:, :, color])
        image_ar = du.format_img_ar(image_ar)
        img.fromarray(image_ar).save(path + file[:-4] + '_gx_gy_mean3_sep.jpg')

    # sf stands for second filter
    sf = np.ones((3, 3, 3), dtype='int64')
    sf[:, :, 0] = F.kernels['gx']
    sf[:, :, 1] = F.kernels['gx']
    sf[:, :, 2] = F.kernels['gx']

    for file in files:
        image_ar = np.int64(img.open(path + file))
        result = du.convolution(du.padding(image_ar, sf, extension=True), sf)
        result = du.format_img_ar(result)
        img.fromarray(result).save(path + file[:-4] + '_gx_gx_gx.jpg')

    for file in files:
        image_ar = np.int64(img.open(path + file))
        padded = du.padding(image_ar, sf[:, :, 0], extension=True)
        for color in range(3):
            image_ar[:, :, color] = du.convolution(padded[:, :, color], sf[:, :, color])
        image_ar = du.format_img_ar(image_ar)
        img.fromarray(image_ar).save(path + file[:-4] + '_gx_gx_gx_sep.jpg')
