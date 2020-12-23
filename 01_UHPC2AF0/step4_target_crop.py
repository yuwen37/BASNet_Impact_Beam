# import the necessary packages
import matplotlib.pyplot as plt
import cv2
import os
os.chdir("E:\\BASNet_Impact_Beam\\")
video_name = '2AF0'

# 设置初始化参数
index = 0

# 获取视频基本信息
vs = cv2.VideoCapture("E://视觉数据/冲击试验高速摄像/UHPC{}.hsv".format(video_name))
fps = int(round(vs.get(cv2.CAP_PROP_FPS)))  # 获取帧率
frame_counter = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取总帧数
duration = frame_counter / fps  # 计算视频时长

# 过滤前期无用帧
for img in range(130*fps):  # 130秒时有撞击
	frame = vs.read()

while True:
    frame = vs.read()
    frame = frame[1]
    index += 1
    if index > 300:
        break
    frame = frame[400:470, 200:800, :]  # H*W*C
    # plt.imshow(frame)
    # plt.show()

    h, w = frame.shape[:2]
    subpixel_level = 4
    frame = cv2.resize(frame, (w * subpixel_level, h * subpixel_level), None, 0, 0, cv2.INTER_CUBIC)
    # 增强亮度
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # BGR2HSV
    frame[:, :, 2] = 1 * frame[:, :, 2]  # 亮度不能太大，否则会线条识别断裂
    frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
    if not os.path.exists('./test_data/UHPC{}/target_ROI'.format(video_name)):
        os.makedirs('./test_data/UHPC{}/target_ROI'.format(video_name))
    cv2.imwrite('./test_data/UHPC{}/target_ROI/UHPC{}_{}.jpg'.format(video_name, video_name, str(index)), frame)
print('./test_data/UHPC{}/target_ROI/UHPC{}_{}.jpg'.format(video_name, video_name, str(index)))
