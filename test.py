import pandas as pd

# 读取csv文件
test_df = pd.read_csv('data/archive/Test.csv')
labels_df = pd.read_csv('data/archive/20bn-jester-download-package-labels/labels/test-labels.csv')

# 合并数据
merged_df = test_df.merge(labels_df, how='left', left_on='id', right_on='video_id')

# 删除不需要的列
merged_df.drop(columns=['label_x', 'label_id_x', 'video_id'], inplace=True)

# 重命名列
merged_df.rename(columns={'label_y': 'label', 'label_id_y': 'label_id'}, inplace=True)

# 保存合并后的数据
merged_df.to_csv('data/archive/Test_new.csv', index=False)

print("数据已成功合并并保存到Test_new.csv文件中。")
