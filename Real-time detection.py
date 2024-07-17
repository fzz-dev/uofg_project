import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
import torch
from HandGestureLSTM import HandGestureLSTM


# 初始化 MediaPipe 手部模型
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)


# 定义关键点提取函数
def extract_keypoints(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = mp_hands.process(image_rgb)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
        return keypoints
    else:
        return np.zeros(21 * 3)  # 如果没有检测到手，返回零向量


# 标签编码
labels = pd.read_csv('data/archive/Train.csv')['label'].unique()
label_map = {label: idx for idx, label in enumerate(labels)}
label_map = {v: k for k, v in label_map.items()}

# 模型参数
input_size = 21 * 3
hidden_size = 64
output_size = len(label_map)

# 初始化并加载模型
model = HandGestureLSTM(input_size, hidden_size, output_size)
model.load_state_dict(torch.load('best_model (2).pth', map_location=torch.device('cpu')))
model.eval()

# 初始化摄像头
cap = cv2.VideoCapture(0)

# 设置帧缓存和缓冲区大小
frame_buffer = []
buffer_size = 30  # 例如缓冲区大小为 30 帧

# 实时检测
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    keypoints = extract_keypoints(frame)
    frame_buffer.append(keypoints)
    if len(frame_buffer) > buffer_size:
        frame_buffer.pop(0)

    if len(frame_buffer) == buffer_size:
        input_tensor = torch.tensor([frame_buffer], dtype=torch.float32)
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            gesture = label_map[predicted.item()]
        cv2.putText(frame, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
