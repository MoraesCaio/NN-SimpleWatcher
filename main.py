import argparse
from pathlib import Path
from PIL import Image as img
import numpy as np
import deep_utils as du
import keras


def file_exist(path_str, verbose=False):
    exist = Path(path_str).is_file()
    if not exist and verbose:
        print('This file \"' + path_str + '\" does not exist.')
    return exist


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--image', type=str, default='', help='Image file.')
    parser.add_argument('-zp', '--zero-padding', type=bool, default=False, help='Zero padding.')
    parser.add_argument('-k1', '--kernel1', type=str, default='', help='Convolution mask file.')
    parser.add_argument('-k2', '--kernel2', type=str, default='', help='Convolution mask file.')
    parser.add_argument('-k3', '--kernel3', type=str, default='', help='Convolution mask file.')
    parser.add_argument('-b1', '--bias1', type=str, default='', help='Convolution bias file.')
    parser.add_argument('-b2', '--bias2', type=str, default='', help='Convolution bias file.')
    parser.add_argument('-b3', '--bias3', type=str, default='', help='Convolution bias file.')
    parser.add_argument('-s', '--stride', type=int, default=1, help='Convolution mask stride.')
    parser.add_argument('-fa1', '--activation-function1', type=str, default='', help='Activation function. (\'\', relu or tanh)')
    parser.add_argument('-fa2', '--activation-function2', type=str, default='', help='Activation function. (\'\', relu or tanh)')
    parser.add_argument('-fa3', '--activation-function3', type=str, default='', help='Activation function. (\'\', relu or tanh)')
    parser.add_argument('-n', '--normalize', action="store_true", help='Normalize.')
    parser.add_argument('-v', '--view', type=bool, default=True, help='View image.')
    parser.add_argument('-o', '--output', type=str, default='', help='Output file.')

    args, _ = parser.parse_known_args()

    # FOR TESTING ON IDES
    # args.image = 'd.jpg'
    # args.zero_padding = True
    # args.kernel1 = 'k1.txt'
    # args.kernel2 = 'k2.txt'
    # args.kernel3 = 'k3.txt'
    # args.bias1 = 'b1.txt'
    # args.bias2 = 'b2.txt'
    # args.bias3 = 'b3.txt'
    # args.stride = 1
    # args.activation_function1 = 'relu'
    # args.activation_function2 = 'relu'
    # args.activation_function3 = 'relu'
    # args.normalize = True
    # args.view = True
    # args.output = 'k1k2k3.jpg'

    image_ar = None

    if not file_exist(args.image, verbose=True):
        return
    else:
        image_ar = np.array(img.open(args.image))

    # KERNELS AND BIASSES

    kernels = []
    biasses = []

    # Kernel1
    if not file_exist(args.kernel1, verbose=True):
        return

    k1 = du.file_to_ar3d(args.kernel1)
    kernels.append(k1)

    # Bias1
    if file_exist(args.bias1):
        with open(args.bias1) as f:
            biasses.append(np.float64(f.readline()))
    else:
        biasses.append(np.float64(0.0))

    # Kernel2 and Kernel3
    if file_exist(args.kernel2) and file_exist(args.kernel3):
        k2 = du.file_to_ar3d(args.kernel2)
        k3 = du.file_to_ar3d(args.kernel3)

        if k1.shape != k2.shape \
                or k1.shape != k3.shape \
                or k2.shape != k3.shape:
            print('Incompatible kernel shapes. Each kernel must have same shape.')
            return

        kernels.extend([k2, k3])

        # Bias2 and Bias3
        b2b3 = [args.bias2, args.bias3]
        for b in b2b3:
            if file_exist(b):
                with open(b) as f:
                    biasses.append(np.float64(f.readline()))
            else:
                biasses.append(np.float64(0.0))

    # PADDING

    outs = []
    if args.zero_padding:
        for k, b in zip(kernels, biasses):
            outs.append(du.padding(image_ar, k))

    # CONVOLUTIONS

    for o, k, b in zip(range(len(outs)), kernels, biasses):
        outs[o] = du.convolution(outs[o], k, stride=args.stride, bias=b)

    # ACTIVATION FUNCTIONS

    fas = [args.activation_function1, args.activation_function2, args.activation_function3]
    for o, fa in zip(range(len(outs)), fas):
        if fa == 'relu':
            outs[o] = du.relu(outs[o])
        elif fa == 'tanh':
            outs[o] = du.tanh(outs[o])

    # NORMALIZATION

    if args.normalize:
        for o in range(len(outs)):
            outs[o] = du.normalize_uint8(outs[o])

    # RESULT

    result = None
    if len(outs) == 3:
        result = np.zeros((outs[0].shape[0], outs[0].shape[1], 3), dtype='uint8')
        for ch in range(3):
            result[:, :, ch] = du.format_img_ar(outs[ch])
    elif len(outs) == 1:
        result = du.format_img_ar(outs[0])
    else:
        print('Invalid number of channels:', len(outs))
        return

    # OUTPUT (VIEW AND SAVING)

    result_img = img.fromarray(result, mode='L' if len(outs) == 1 else 'RGB')
    if args.view:
        img.open(args.image).show()
        result_img.show('Result')
    if args.output:
        result_img.save(args.output)


if __name__ == '__main__':
    main()
