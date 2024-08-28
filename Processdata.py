import pandas as pd
import os
import glob

# 读取上传的CSV文件
test_df = pd.read_csv('data/archive/Test_new.csv')
train_df = pd.read_csv('data/archive/Train.csv')
validation_df = pd.read_csv('data/archive/Validation.csv')

# 定义需要删除的标签和新的标签映射
labels_to_delete = [
    'Rolling Hand Forward', 'Rolling Hand Backward', 'Turning Hand Clockwise', 'Turning Hand Counterclockwise',
    'No gesture', 'Drumming Fingers'
]
new_label_mapping = {'Doing other things': 'No gesture'}


# 更新数据框的函数
def update_dataframe(df):
    # 删除标签为“No gesture”的行
    df = df[~df['label'].isin(labels_to_delete)]
    # 更新标签
    df['label'] = df['label'].replace(new_label_mapping)
    # 重新分配标签ID
    df['label_id'] = pd.factorize(df['label'])[0]
    return df


# 更新所有数据框
updated_test_df = update_dataframe(test_df)
updated_train_df = update_dataframe(train_df)
updated_validation_df = update_dataframe(validation_df)

# 保存更新后的数据框为新的CSV文件
updated_test_df.to_csv('data/archive/Updated_Test_new.csv', index=False)
updated_train_df.to_csv('data/archive/Updated_Train.csv', index=False)
updated_validation_df.to_csv('data/archive/Updated_Validation.csv', index=False)


# 删除相应的npy文件的函数
def delete_npy_files(df, folder_path):
    for _, row in df.iterrows():
        video_id = row['video_id']
        label = row['label']
        if label in labels_to_delete:
            # 根据video_id构建npy文件的路径
            npy_pattern = os.path.join(folder_path, f"{video_id}.npy")
            for npy_file in glob.glob(npy_pattern):
                os.remove(npy_file)


# 假设npy文件存放在 /mnt/data/npy_files/ 目录下
train_folder_path = 'data/processed_train/'
train_aug = 'data/processed_train_with_aug/'
test_folder_path = 'data/processed_test/'
val_folder_path = 'data/processed_validation/processed_validation/'

# 删除相应的npy文件
delete_npy_files(test_df, test_folder_path)
delete_npy_files(train_df, train_folder_path)
delete_npy_files(train_df, train_aug)
delete_npy_files(validation_df, val_folder_path)

labels = pd.read_csv('data/archive/Updated_Train.csv')['label'].unique()
label_map = {label: idx for idx, label in enumerate(labels)}
print(label_map)
print("CSV文件已更新并保存，npy文件已删除。")

