import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
plt.rcParams['font.sans-serif']=['Microsoft YaHei']
os.chdir("E:\\BASNet_working_condition\\")
video_name = '3AF0'

disp_opencv = []
disp_trad = []
disp_bas = []
factors = np.loadtxt(fname='./results/UHPC{}/target_displacement/factors.txt'.format(video_name), dtype=float,  unpack=True)
path_trads = './冲击梁答案/Part1/UHPC{}.txt'.format(video_name)
disp_trads = pd.read_csv(path_trads, header=None, names=['时间', '测点3', '测点1', '测点2', '测点4', '测点5'])
if not os.path.exists('./results/UHPC{}/figure'.format(video_name)):
    os.makedirs('./results/UHPC{}/figure'.format(video_name))

for i in range(5):
    # 读入三种测点数据数据
    disp_trad = disp_trads['测点{}'.format(str(i + 1))]
    path_opencv = './results/UHPC{}/target_displacement/target{}_displacement.txt'.format(video_name, str(i + 1))
    disp_opencv = pd.read_csv(path_opencv, header=None, sep=' ', names=['x', 'y'])
    path_bas = './results/UHPC{}/edge/edge{}.txt'.format(video_name, str(i + 1))
    disp_bas = pd.read_csv(path_bas, header=None)

    # 处理数据
    disp_trad = np.array(disp_trad)[17:517]
    disp_opencv = np.array(disp_opencv['y']) * factors
    disp_opencv = disp_opencv - disp_opencv[0]
    disp_bas = np.array(disp_bas).reshape(-1) * factors

    # 绘图
    plt.figure(i, figsize=(16, 9))
    plt.plot(disp_trad, color='blue', marker='o', linewidth=1, markersize=1)
    plt.plot(disp_opencv, color='red', marker='o', linestyle='--', linewidth=1.4, markersize=3)
    plt.plot(disp_bas, color='black', linewidth=1, markersize=1)
    plt.title('测点{}位移时程'.format(i + 1), fontsize=20)
    plt.legend(['Marker Displacment-Software', 'Marker Displacment-OpenCV', 'Edge Displacement'], fontsize=20)
    plt.ylabel('位移', fontsize=20)
    plt.xlabel('时间帧', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.show()
    plt.savefig('./results/UHPC{}/figure/测点{}位移时程.jpg'.format(video_name, str(i + 1)), dpi=1080)
    plt.close(i)
