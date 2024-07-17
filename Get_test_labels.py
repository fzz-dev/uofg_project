import csv

# 定义标签映射字典
label_map = {
    'Doing other things': 0,
    'Pushing Two Fingers Away': 1,
    'Drumming Fingers': 2,
    'Sliding Two Fingers Down': 3,
    'Pushing Hand Away': 4,
    'Shaking Hand': 5,
    'Pulling Two Fingers In': 6,
    'Stop Sign': 7,
    'Zooming In With Two Fingers': 8,
    'Sliding Two Fingers Up': 9,
    'Zooming Out With Two Fingers': 10,
    'Zooming In With Full Hand': 11,
    'No gesture': 12,
    'Swiping Right': 13,
    'Thumb Down': 14,
    'Rolling Hand Forward': 15,
    'Pulling Hand In': 16,
    'Zooming Out With Full Hand': 17,
    'Swiping Left': 18,
    'Rolling Hand Backward': 19,
    'Turning Hand Counterclockwise': 20,
    'Swiping Up': 21,
    'Turning Hand Clockwise': 22,
    'Sliding Two Fingers Left': 23,
    'Swiping Down': 24,
    'Thumb Up': 25,
    'Sliding Two Fingers Right': 26
}

# 读取原CSV文件
input_file_path = 'data/archive/20bn-jester-download-package-labels/labels/test-answers.csv'
output_file_path = 'data/archive/20bn-jester-download-package-labels/labels/test-labels.csv'

# 打开输入和输出文件
with open(input_file_path, 'r', encoding='utf-8') as infile, open(output_file_path, 'w', newline='', encoding='utf-8') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # 写入新的CSV文件的表头
    writer.writerow(['vedio_id', 'label', 'label_id'])

    # 逐行处理输入文件中的数据
    for row in reader:
        if row:  # 忽略空行
            vedio_id, label = row[0].split(';')
            label_id = label_map.get(label, -1)  # 如果找不到标签，使用-1表示未知标签
            writer.writerow([vedio_id, label, label_id])

print("文件处理完成，已保存为:", output_file_path)
