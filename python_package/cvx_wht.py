import numpy as np
from ctypes import cdll
from ctypes import c_int
from ctypes import c_void_p
import cv2 as cv

#@todo hardcode library
lib = cdll.LoadLibrary('/Users/jimmy/Source/opencv_util/build/libcvx_wht_python.dylib')


def extract_WHT_feature(rgb_im, points, patch_size, kernel_num):
    assert len(rgb_im.shape) == 3
    assert rgb_im.shape[2] == 3
    assert rgb_im.dtype == np.uint8
    assert len(points.shape) == 2
    assert points.shape[1] == 2
    assert patch_size == 16 or patch_size == 32 or patch_size == 64 or patch_size == 128


    rows, cols = rgb_im.shape[0], rgb_im.shape[1]
    point_num = points.shape[0]
    dim = kernel_num * 3 - 3
    features = np.zeros((point_num, dim))
    lib.extractWHTFeature(c_void_p(rgb_im.ctypes.data),
                          c_int(rows),
                          c_int(cols),
                          c_int(3),
                          c_void_p(points.ctypes.data),
                          c_int(point_num),
                          c_int(patch_size),
                          c_int(kernel_num),
                          c_void_p(features.ctypes.data))
    return features

def ut_extract_WHT_feature():
    import random
    im = cv.imread('/Users/jimmy/Desktop/test_image.png')
    rows = im.shape[0]
    cols = im.shape[1]
    patch_size = 32
    kernel_num = 20
    points = np.zeros((20, 2))
    for i in range(20):
        points[i, 0] = random.randint(1, cols)
        points[i, 1] = random.randint(1, rows)

    features = extract_WHT_feature(im, points, patch_size, kernel_num)
    print(features.shape)







if __name__ == '__main__':
    ut_extract_WHT_feature()

