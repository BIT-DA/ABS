import json

# 加载 cupl_new.json 和 cub.json 文件
with open('cupl_new.json', 'r') as cupl_file:
    cupl_data = json.load(cupl_file)

with open('/mnt/bit/clc/WCA-main/WCA-main/features/cub/cub.json', 'r') as cub_file:
    cub_keys = json.load(cub_file)

# 确保两个文件的长度匹配
if len(cupl_data) != len(cub_keys):
    raise ValueError("cupl_new.json 和 cub.json 的长度不匹配，无法替换键！")

# 创建新的字典
converted_data = {cub_key: cupl_data[old_key] for cub_key, old_key in zip(cub_keys, cupl_data.keys())}

# 保存到新文件
with open('converted_cupl.json', 'w') as output_file:
    json.dump(converted_data, output_file, indent=4)

print("键替换完成，结果已保存到 converted_cupl.json")
