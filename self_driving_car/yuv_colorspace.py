# rgb2yuv is only avaiable in version 0.13dev of scipy-image
import skimage
import numpy as np
from skimage import color
from scipy import linalg

yuv_from_rgb = np.array([[ 0.299     ,  0.587     ,  0.114      ],
                                                  [-0.14714119, -0.28886916,  0.43601035 ],
                                                  [ 0.61497538, -0.51496512, -0.10001026 ]])

rgb_from_yuv = linalg.inv(yuv_from_rgb)

def rgb2yuv(rgb):
    return skimage.color.colorconv._convert(yuv_from_rgb, rgb)

def yuv2rgb(yuv):
    return skimage.color.colorconv._convert(rgb_from_yuv, yuv)
