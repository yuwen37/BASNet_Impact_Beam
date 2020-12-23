import numpy as np
import cv2
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from skimage import morphology
import os
from scipy import ndimage
from scipy.fftpack import fft, ifft
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
os.chdir("E:\\BASNet_Impact_Beam\\")
video_name = '4AF0'

pd.set_option('precision', 15)
np.set_printoptions(precision=15)
factors = np.loadtxt(fname='./results/UHPC{}/target_displacement/factors.txt'.format(video_name), dtype=float,  unpack=True)
K_t = []
corner_t = []
tx = []
n = len(factors) - 100
for i in range(5):
    corner_t.append(np.zeros(n))
    K_t.append(np.zeros(n))
    target = pd.read_csv('./results/UHPC{}/target_displacement/target{}_displacement.txt'.format(video_name, i + 1), header=None, encoding='gb2312', sep=' ')
    tx.append(np.array(target.loc[:, 0]).astype(np.int32))

for img in range(n):
    BW_path = './test_data/UHPC{}/BW/UHPC{}_{}.png'.format(video_name, video_name, img + 1)
    BW = cv2.imread(BW_path)
    ret, BW = cv2.threshold(BW[:, :, 0], 100, 1, cv2.THRESH_BINARY)
    BW = morphology.remove_small_objects(BW > 0, 40000, connectivity=1)
    BW = BW.astype(np.uint8)

    # 提取边缘方法二
    # contours, hierarchy = cv2.findContours(BW,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) # CHAIN_APPROX_NONE
    contours, hierarchy = cv2.findContours(BW, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    px = np.array(contours[0][:, :, 0]).squeeze()  # W
    py = np.array(contours[0][:, :, 1]).squeeze()  # H

    df = pd.DataFrame({'x': px, 'y': py})
    df = df.groupby(['x']).max()
    df.index.name = 'index'
    df['x'] = df.index
    df['z'] = df['y'].rolling(300, axis=0, min_periods=None, center=True).mean()
    df['y'] = df['z']
    df = df.dropna(axis=0, how='any')

    df.sort_values('x', ascending=True, inplace=True)
    px = np.array(df['x'])
    py = np.array(df['y'])#.astype(np.int32)

    # 图像距离和实际距离换算
    # px = px * factors[img]
    # py = py * factors[img]

    # 间隔点计算转角
    delta = 200
    corner = np.zeros(len(px))
    for i in range(delta, len(px)-delta):
        slope = (py[i-delta]-py[i+delta])/(px[i-delta]-px[i+delta])
        corner[i] = np.arctan(slope)/3.1415926*180
    # 计算各靶点转角
    for i in range(5):
        loc_edge_x = np.where(px == tx[i][img])
        try:
            corner_t[i][img] = corner[loc_edge_x]
        except:
            corner_t[i][img] = corner_t[i][-1]

    # 间隔点计算曲率
    delta = 400
    K = np.zeros(len(px))
    for i in range(delta, len(px) - delta):
        n1 = (px[i] - px[i - delta], py[i] - py[i - delta])
        n2 = (-px[i] + px[i + delta], -py[i] + py[i + delta])
        L = np.sqrt((px[i + delta] - px[i - delta]) ** 2 + (py[i + delta] - py[i - delta]) ** 2)
        H = np.abs(n1[0] * n2[1] - n1[1] * n2[0]) / L
        K[i] = 2 * H / (H ** 2 + (L / 2) ** 2) / (factors[img]/1000)
    # 计算各靶点曲率
    for i in range(5):
        loc_edge_x = np.where(px == tx[i][img])
        try:
            K_t[i][img] = K[loc_edge_x]
        except:
            K_t[i][img] = K_t[i][-1]

if not os.path.exists('./results/UHPC{}/corner'.format(video_name)):
    os.makedirs('./results/UHPC{}/corner'.format(video_name))
if not os.path.exists('./results/UHPC{}/curve'.format(video_name)):
    os.makedirs('./results/UHPC{}/curve'.format(video_name))
for i in range(5):
    np.savetxt('./results/UHPC{}/corner/corner{}.txt'.format(video_name, i + 1), corner_t[i], fmt='%.10f')
    np.savetxt('./results/UHPC{}/curve/curve{}.txt'.format(video_name, i + 1), K_t[i], fmt='%.10f')