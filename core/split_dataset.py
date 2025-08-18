import random
import os

# --- 配置 ---
# 输入：我们已经清理好的、包含4635行有效数据的清单文件
input_annotation_file = './data/dataset_fog/voc_norm_train.txt'

# 输出：两个全新的清单文件
new_train_file = './data/dataset_fog/train_final.txt'  # 最终的训练集清单
new_val_file = './data/dataset_fog/val_final.txt'      # 最终的验证集清单

# 设置验证集占总数据的比例
validation_split_ratio = 0.1 # 我们取大约10%作为验证集，接近8:1的比例

# --- 主程序 ---
print("开始划分数据集...")

try:
    with open(input_annotation_file, 'r') as f:
        lines = f.readlines()

    # 随机打乱所有数据行，确保划分是无偏的
    random.shuffle(lines)

    # 计算划分点
    num_total = len(lines)
    num_validation = int(num_total * validation_split_ratio)
    num_training = num_total - num_validation
    
    print(f"总共有 {num_total} 条有效数据。")
    print(f"将其划分为: {num_training} 条训练数据 和 {num_validation} 条验证数据。")

    # 分割数据
    train_lines = lines[:num_training]
    val_lines = lines[num_training:]

    # 写入新的训练集文件
    with open(new_train_file, 'w') as f_train:
        f_train.writelines(train_lines)
    print(f"成功创建最终训练集清单: {new_train_file}")

    # 写入新的验证集文件
    with open(new_val_file, 'w') as f_val:
        f_val.writelines(val_lines)
    print(f"成功创建最终验证集清单: {new_val_file}")

except FileNotFoundError:
    print(f"错误：找不到输入的清单文件 '{input_annotation_file}'。请先运行 clean_annotation.py。")
