'''
@Autuor: LZ-CH
@Contact: 2443976970@qq.com
'''

import numpy as np
import cv2
from PIL import Image
import numpy as np
import os


def lplas_decomposition(img, level_num=3):
    #input's type: <numpy>
    #output's type: <list<numpy>>
    # # 检查输入图像的尺寸是否满足要求
    # if img.shape[0] <= 0 or img.shape[1] <= 0:
    #     raise ValueError("Input image dimensions must be greater than zero.")
    G_list = []
    L_list = []
    G_list.append(img)
    for i in range(level_num-1):
        # print('[lpls de 21] G_list[{}], {}'.format(i, G_list[i]))
        # print('[lpls de 22] G_list[{}], {}'.format(i, G_list[i].shape))
        G_list.append(cv2.pyrDown(G_list[i]))
    for j in range(level_num-1):
        L_list.append(G_list[j]-cv2.pyrUp(G_list[j+1], dstsize=(G_list[j].shape[1], G_list[j].shape[0])))
    L_list.append(G_list[level_num-1])
    G_list.reverse()
    L_list.reverse()
    return G_list, L_list  # G_list里放的是高斯金字塔， L_list放的是拉普拉斯金字塔
if __name__ =='__main__':
    # print('path:', os.getcwd())
    img =cv2.imread('491.png')
    # img = Image.open('35.jpg')
    img_arry = np.array(img) / 255
    print('[lpls 37] img_arry.shape',img_arry.shape)


    # print(img.shape)
    # img = img/255.0
    g, L = lplas_decomposition(img_arry)

    cv2.imwrite('g0.jpg', g[0] * 255)
    cv2.imwrite('g1.jpg', g[1] * 255)
    cv2.imwrite('g2.jpg', g[2] * 255)
    cv2.imwrite('l0.jpg', L[0] * 255) #全局色彩信息 低分辨率
    cv2.imwrite('l1.jpg', L[1] * 255) #细节信息 高解析度
    cv2.imwrite('l2.jpg', L[2] * 255)
    # print('[lpls 50] l.shape', L[0].shape,L[1].shape,L[2].shape)
    # print('[lpls 50] g.shape', g[0].shape, g[1].shape, g[2].shape)
    # newshape1 = (400,600,3)
    # newshape2 = (400, 600,3)
    # re1 = np.resize(L[0], newshape1)
    # re2 = np.resize(L[1], newshape2)
    # cv2.imwrite('plus.jpg', (re1 * 255)+(re2 * 255)+(L[2] * 255) )