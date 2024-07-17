import cv2
import os
import pandas as pd
import time
from datetime import datetime

# 创建数据保存的主目录
data_dir = "captured_actions"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# 打开摄像头
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open the webcam")
    exit()

# 设置标签
labels = []
action_count = 0
images_per_action = 27
capture_interval = 0.1  # 每次采集之间的间隔时间，单位：秒

print("Press 'q' to quit the data collection.")
print("Press 'a' to start collecting a new action.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read the frame")
        break

    # 显示当前捕获的图像
    cv2.imshow('Frame', frame)

    # 等待键盘输入
    key = cv2.waitKey(1) & 0xFF

    # 按'q'键退出
    if key == ord('q'):
        break

    # 按'a'键开始一个新的动作采集
    if key == ord('a'):
        action_count += 1
        current_label = input("Enter label for the new action: ")
        action_dir = os.path.join(data_dir, f"action_{action_count}_{current_label}")
        os.makedirs(action_dir, exist_ok=True)
        print(f"Started new action {action_count} with label {current_label}")

        # 等待一秒钟
        print("Starting in 1 second...")
        time.sleep(1)

        # 开始采集27张照片
        start_time = datetime.now()
        for image_count in range(images_per_action):
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read the frame")
                break

            img_name = os.path.join(action_dir, f"image_{image_count}.png")
            cv2.imwrite(img_name, frame)
            print(f"Captured {img_name} for action {current_label}")

            # 等待一段时间再采集下一张照片
            time.sleep(capture_interval)

        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()
        print(f"Finished collecting action {current_label} in {elapsed_time:.2f} seconds")

        # 添加文件夹及其对应的标签
        labels.append({'folder': action_dir, 'label': current_label})

# 释放摄像头并关闭所有窗口
cap.release()
cv2.destroyAllWindows()

# 保存文件夹及标签到CSV文件
labels_df = pd.DataFrame(labels)
labels_file = os.path.join(data_dir, "labels.csv")
labels_df.to_csv(labels_file, index=False)
print(f"Saved labels to {labels_file}")
