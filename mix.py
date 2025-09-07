import json
import random

# 文件路径
file1_path = 'data/original/api25.json'
file2_path = 'data/original/api26.json'
count = 500

random.seed(42)

# 读取两个文件
with open(file1_path, 'r', encoding='utf-8') as f1, open(file2_path, 'r', encoding='utf-8') as f2:
    data1 = json.load(f1)
    data2 = json.load(f2)

print(len(data1))
print(len(data2))
# 确保是列表类型
if not isinstance(data1, list) or not isinstance(data2, list):
    raise ValueError("两个 JSON 文件的内容应为列表格式")

# 选取 file2 中的 20% 数据
sampled_data = random.sample(data2, count)
print(len(sampled_data))

# 从 file2 中删除这部分数据
data2 = [item for item in data2 if item not in sampled_data]
print(len(data2))

# 把 sampled_data 混入 file1 中（这里是添加到末尾）
data1.extend(sampled_data)
print(len(data1))

# 写回文件
with open(file1_path, 'w') as file:
    file.write(json.dumps(data1))

with open(file2_path, 'w') as file:
    file.write(json.dumps(data2))
