import os
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp


# 初始化 MediaPipe 手部模型
mp_hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)

# 定义关键点提取函数
def extract_keypoints(image):
    if image is None:
        print('图像不存在')
        return np.zeros(21 * 3)  # 如果没有读取到图像，返回零向量
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = mp_hands.process(image_rgb)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
        return keypoints
    else:
        return np.zeros(21 * 3)  # 如果没有检测到手，返回零向量


# 数据预处理并缓存到文件
def preprocess_and_cache(csv_file, data_dir, output_dir):
    df = pd.read_csv(csv_file)
    os.makedirs(output_dir, exist_ok=True)
    for idx, row in df.iterrows():
        video_id = row['video_id']
        frame_count = row['frames']
        frames = []
        for i in range(1, frame_count + 1):
            img_path = os.path.normpath(os.path.join(data_dir, str(video_id), f'{i:05d}.jpg'))
            img = cv2.imread(img_path)
            keypoints = extract_keypoints(img)
            frames.append(keypoints)
        frames = np.array(frames)
        np.save(os.path.join(output_dir, f'{video_id}.npy'), frames)


def preprocess_and_cache_test(csv_file, data_dir, output_dir):
    df = pd.read_csv(csv_file)
    os.makedirs(output_dir, exist_ok=True)
    for idx, row in df.iterrows():
        video_id = row['id']
        frame_count = row['frames']
        frames = []
        for i in range(1, frame_count + 1):
            img_path = os.path.normpath(os.path.join(data_dir, str(video_id), f'{i:05d}.jpg'))
            img = cv2.imread(img_path)
            keypoints = extract_keypoints(img)
            frames.append(keypoints)
        frames = np.array(frames)
        np.save(os.path.join(output_dir, f'{video_id}.npy'), frames)


# 示例调用
# preprocess_and_cache('data/archive/Train.csv', 'data/archive/Train', 'data/processed_train')
# preprocess_and_cache('data/archive/Validation.csv', 'data/archive/Validation', 'data/processed_validation')
preprocess_and_cache_test('data/archive/Test.csv', 'data/archive/Test', 'data/processed_test')
