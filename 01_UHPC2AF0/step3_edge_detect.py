# import the necessary packages
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
os.chdir("E:\\BASNet_Impact_Beam\\")
video_name = '2AF0'

for img in range(500):
    img_path = './test_data/UHPC{}/test_images/UHPC{}_{}.jpg'.format(video_name, video_name, str(img + 1))
    label_path = './test_data/UHPC{}/test_results/UHPC{}_{}.png'.format(video_name, video_name, str(img + 1))
    if not os.path.exists('./test_data/UHPC{}/BW'.format(video_name)):
        os.makedirs('./test_data/UHPC{}/BW'.format(video_name))
    BW_path = './test_data/UHPC{}/BW/UHPC{}_{}.png'.format(video_name, video_name, str(img + 1))
    img = cv2.imread(img_path)
    label = cv2.imread(label_path)
    label = cv2.cvtColor(label, cv2.COLOR_RGB2GRAY)

    # 平滑边缘
    # kernel = np.ones((5, 5), np.float32) / 25
    # label = cv2.filter2D(label, -1, kernel)
    label = cv2.GaussianBlur(label, (51, 51), 0)

    ret, BW = cv2.threshold(label, 40, 1, cv2.THRESH_BINARY)
    kernel = np.ones((1, 50))
    BW = cv2.morphologyEx(BW, cv2.MORPH_OPEN, kernel)

    # 输出处理后二值图
    cv2.imwrite(BW_path, BW*255)
    # edge_y, edge_x = np.where(BW == 1)
    # df = pd.DataFrame({'x': edge_x, 'y': edge_y})
    # data_group = df.groupby(['x']).max()
    # data_group.index.name = 'index'
    # data_group['x'] = data_group.index
    # data_group.sort_values('x', ascending=True, inplace=True)
    #
    # data_group['y'] = data_group['y'].rolling(window=50, min_periods=1, center=True).mean()
    # edge_x = np.array(data_group['x']).astype(np.int)
    # edge_y = np.array(data_group['y']).astype(np.int)
    # plt.figure(img)
    # plt.plot(edge_x[500:-1], edge_y[500:-1], c='r',  label='edge', linewidth=0.5)
    # plt.title('Frame' + str(img))
    # plt.savefig('./test_data/edge_line/第' + str(img) + '帧.jpg', dpi=1080)
    # plt.close()

    # img[y, x, :] = [0, 0, 230]
    # h, w = img.shape[:2]
    # cv2.namedWindow('VIDEO', 0)
    # cv2.resizeWindow('VIDEO', w, h)
    # cv2.imshow('VIDEO', img)
    # cv2.waitKey(100)
# cv2.destroyAllWindows()
