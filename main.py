# import dip
import numpy as np
from PIL import Image
import cv2
import p


# a = Image.open('a.jpg')
# # a = a.convert(mode='L')
# # a.save('bw.jpg')
# bw = Image.open('bw.jpg')

# print('a = ', np.array(a).shape)
# print('bw = ', np.array(bw).shape)

def show(ar):
    if len(ar.shape) == 3:
        print('RGB')
        Image.fromarray(cv2.cvtColor(ar, cv2.COLOR_BGR2RGB) , 'RGB').show()
    elif len(ar.shape) == 4:
        print('RGBA')
        Image.fromarray(ar, 'RGBA').show()
    else:
        print('L')
        Image.fromarray(ar, 'L').show()
    input('press enter')

# sobel_gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
# a_cv = cv2.imread('a.jpg', -1)
# print(a_cv.shape)
# show(a_cv)
# padded = p.padding(a_cv, np.zeros((29, 29)))
# show(padded)
# conv = p.run_kernel_loop(padded, sobel_gx)

qq = np.array([[[1,1,1],\
                [1,1,1], # cor pixel (0,1)
                [1,1,1]],\
               [[2,2,2],\
                [2,2,2],\
                [2,2,2]],\
               [[3,3,3],\
                [3,3,3],\
                [3,3,3]]]) #30 -30 40
ww = np.array([[[1,1,1],\
                [1,1,1],\
                [1,1,1]],\
               [[1,1,1],\
                [1,1,1],\
                [1,1,1]],\
               [[1,1,1],\
                [1,1,1],\
                [1,1,1]]])
w = np.array([[1,1,1],\
              [1,1,1],\
              [1,1,1]])

print(p.convolution(qq, ww))

# for i in range(-1, 2):
#     print('OPTION: ' + str(i))
#     a_cv = cv2.imread('a.jpg', i)
#     bw_cv = cv2.imread('bw.jpg', i)

    # print('a_cv = ', a_cv.shape)
    # input()
    # if len(a_cv.shape) == 3:
    #     Image.fromarray(cv2.cvtColor(a_cv, cv2.COLOR_BGR2RGB)).show()
    # else:
    #     Image.fromarray(a_cv).show()

    # print('bw_cv = ', bw_cv.shape)
    # input()
    # if len(bw_cv.shape) == 3:
    #     Image.fromarray(cv2.cvtColor(bw_cv, cv2.COLOR_BGR2RGB)).show()
    # else:
    #     Image.fromarray(bw_cv).show()

# -1 número de canais corresponde ao real: p&b - 1 canal, rgb - 3 canais
#  0 só 1 canal; força p&b
#  1 sempre 3 canais


