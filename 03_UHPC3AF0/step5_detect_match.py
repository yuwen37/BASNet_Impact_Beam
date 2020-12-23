# 第一步: 计算像素点实际距离
# 第二步：计算边缘和靶点的对应横坐标
# 第三步：计算靶点位移和边缘位移
import numpy as np
from PIL import Image
import glob
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from skimage import morphology
from scipy import ndimage
import os
plt.rcParams['font.sans-serif']=['Microsoft YaHei']
os.chdir("E:\\BASNet_working_condition\\")
video_name = '3AF0'

target_y_displacement = [[] for _ in range(5)]
target_x_displacement = [[] for _ in range(5)]
edge_y = [[] for _ in range(5)]  # 测点对应的边缘y方向变化 
factors = []

for img in range(500):
    label_path = './test_data/UHPC{}/test_results/UHPC{}_{}.png'.format(video_name, video_name, str(img + 1))
    target_path = './test_data/UHPC{}/target_ROI/UHPC{}_{}.jpg'.format(video_name, video_name, str(img + 1))
    BW_path = './test_data/UHPC{}/BW/UHPC{}_{}.png'.format(video_name, video_name, str(img + 1))

    label = cv2.imread(label_path)
    label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    ret, BW = cv2.threshold(label, 75, 1, cv2.THRESH_BINARY)

    # STEP1 计算像素点实际距离
    ROI = BW[100:900, 1200:2000]
    # plt.figure()
    # plt.imshow(ROI, cmap='binary')
    # plt.show()
    # plt.close()
    y, x = np.where(ROI == 1)
    df = pd.DataFrame({'x': x, 'y': y})
    data_group_up = df.groupby(['x']).max()
    x_up = np.array(data_group_up.index).astype(np.int32)
    y_up = np.array(data_group_up['y']).astype(np.int32)
    data_group_down = df.groupby(['x']).min()
    x_down = np.array(data_group_down.index).astype(np.int32)
    y_down = np.array(data_group_down['y']).astype(np.int32)

    f1 = np.polyfit(x_up, y_up, 1)
    f2 = np.polyfit(x_down, y_down, 1)
    p1 = np.poly1d(f1)
    p2 = np.poly1d(f2)
    line1_y0 = p1(1600)
    line2_y0 = p2(1600)
    factor = 300 / abs(line2_y0 - line1_y0)  # 梁截面高度是300mm
    factors.append(factor)

    # STEP2 计算边缘和靶点的对应横坐标
    target = cv2.imread(target_path)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    # ret, target = cv2.threshold(target, 160, 1, cv2.THRESH_BINARY)
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (20, 20))
    # target = cv2.morphologyEx(target, cv2.MORPH_CLOSE, kernel)
    # plt.figure()
    # plt.imshow(target, cmap='binary')
    # plt.show()
    # plt.close()
    target_point = [np.zeros(target.shape) for _ in range(5)]

    # 这里不能用循环，只能一个一个点单独处理
    target_point[0][:, 350:500] = target[:, 350:500]
    ret, target_point[0] = cv2.threshold(target_point[0], 160, 1, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (20, 20))
    target_point[0] = cv2.morphologyEx(target_point[0], cv2.MORPH_CLOSE, kernel)
    target_point[0] = morphology.remove_small_objects(target_point[0] > 0, 400, connectivity=1)
    # plt.figure()
    # plt.imshow(target_point[0], cmap='binary')
    # plt.show()
    # plt.close()

    target_point[1][:, 700:1000] = target[:, 700:1000]
    ret, target_point[1] = cv2.threshold(target_point[1], 160, 1, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (20, 20))
    target_point[1] = cv2.morphologyEx(target_point[1], cv2.MORPH_CLOSE, kernel)
    target_point[1] = morphology.remove_small_objects(target_point[1] > 0, 400, connectivity=1)
    # plt.figure()
    # plt.imshow(target_point[1], cmap='binary')
    # plt.show()
    # plt.close()

    target_point[2][:, 1150:1350] = target[:, 1150:1350]
    ret, target_point[2] = cv2.threshold(target_point[2], 100, 1, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (30, 30))
    target_point[2] = cv2.morphologyEx(target_point[2], cv2.MORPH_CLOSE, kernel)
    target_point[2] = morphology.remove_small_objects(target_point[2] > 0, 200, connectivity=1)
    # plt.figure()
    # plt.imshow(target_point[2], cmap='binary')
    # plt.show()
    # plt.close()

    target_point[3][:, 1600:1800] = target[:, 1600:1800]
    ret, target_point[3] = cv2.threshold(target_point[3], 100, 1, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (20, 20))
    target_point[3] = cv2.morphologyEx(target_point[3], cv2.MORPH_CLOSE, kernel)
    target_point[3] = morphology.remove_small_objects(target_point[3] > 0, 200, connectivity=1)
    # plt.figure()
    # plt.imshow(target_point[3], cmap='binary')
    # plt.show()
    # plt.close()

    target_point[4][:, 2000:2250] = target[:, 2000:2250]
    ret, target_point[4] = cv2.threshold(target_point[4], 80, 1, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (30, 30))
    target_point[4] = cv2.morphologyEx(target_point[4], cv2.MORPH_CLOSE, kernel)
    target_point[4] = morphology.remove_small_objects(target_point[4] > 0, 200, connectivity=1)
    # plt.figure()
    # plt.imshow(target_point[4], cmap='binary')
    # plt.show()
    # plt.close()

    # 计算各靶点x,y坐标
    tx, ty = [0] * 5, [0] * 5
    for i in range(5):
        ty[i], tx[i] = np.where(target_point[i] == 1)
        tx[i] = np.mean(tx[i])
        ty[i] = np.mean(ty[i])
        target_x_displacement[i].append(tx[i])
        target_y_displacement[i].append(ty[i])

    # STEP3 计算靶点位移和边缘位移
    BW = cv2.imread(BW_path)
    ret, BW = cv2.threshold(BW[:, :, 0], 80, 1, cv2.THRESH_BINARY)
    y, x = np.where(BW == 1)
    df = pd.DataFrame({'x': x, 'y': y})
    df = df.groupby(['x']).max()
    df.index.name = 'index'
    df['x'] = df.index
    df['z'] = df['y'].rolling(100, axis=0, min_periods=None, center=True).mean()
    df['y'] = df['z']
    df = df.dropna(axis=0, how='any')
    x = np.array(df['x']).astype(np.int)
    y = np.array(df['y']).astype(np.int)
    edge_index = [0] * 5
    for i in range(5):
        edge_index[i] = np.where(x == np.int(tx[i]))
        edge_y[i].append(y[edge_index[i]])


if not os.path.exists('./results/UHPC{}/target_displacement'.format(video_name)):
    os.makedirs('./results/UHPC{}/target_displacement'.format(video_name))
if not os.path.exists('./results/UHPC{}/edge'.format(video_name)):
    os.makedirs('./results/UHPC{}/edge'.format(video_name))

np.savetxt('./results/UHPC{}/target_displacement/factors.txt'.format(video_name), factors, fmt='%.10f')
for i in range(5):
    target_y_displacement[i] = np.array(target_y_displacement[i])
    target_x_displacement[i] = np.array(target_x_displacement[i])
    target_displacement = np.concatenate((target_x_displacement[i].reshape((-1, 1)), target_y_displacement[i].reshape((-1, 1))), axis=1)
    np.savetxt('./results/UHPC{}/target_displacement/target{}_displacement.txt'.format(video_name, str(i + 1)), target_displacement, fmt='%.10f')

    edge_y[i] = (np.array(edge_y[i]) - edge_y[i][0])
    np.savetxt('./results/UHPC{}/edge/edge{}.txt'.format(video_name, str(i + 1)), edge_y[i], fmt='%.10f')