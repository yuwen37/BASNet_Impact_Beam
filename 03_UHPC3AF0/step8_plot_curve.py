import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
os.chdir("E:\\BASNet_working_condition\\")
video_name = '3AF0'

corner = []
K = []
for i in range(5):
    corner.append(pd.read_csv('./results/UHPC{}/corner/corner{}.txt'.format(video_name, str(i + 1)), header=None))
    plt.figure(i, figsize=(16, 9))
    plt.plot(corner[i], color='blue', marker='o',  linewidth=1, markersize=3)
    plt.title('测点{}转角时程'.format(i + 1), fontsize=20)
    plt.ylabel('角度', fontsize=20)
    plt.xlabel('时间帧', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig('./results/UHPC{}/figure/测点{}转角时程.jpg'.format(video_name, str(i + 1)), dpi=720)
    # plt.show()
    plt.close(i)

    K.append(pd.read_csv('./results/UHPC{}/curve/curve{}.txt'.format(video_name, str(i + 1)), header=None))
    plt.figure(i, figsize=(16, 9))
    plt.plot(K[i], color='blue', marker='o', linewidth=1, markersize=3)
    plt.title('测点{} 曲率时程'.format(i + 1), fontsize=20)
    plt.ylabel('曲率', fontsize=20)
    plt.xlabel('时间帧', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig('./results/UHPC{}/figure/测点{}曲率时程.jpg'.format(video_name, str(i + 1)), dpi=720)
    # plt.show()
    plt.close(i)

