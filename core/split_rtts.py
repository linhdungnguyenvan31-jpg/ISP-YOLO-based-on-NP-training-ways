import random
import os
from tqdm import tqdm

# --- 1. 请在这里配置您的文件路径 ---
# 输入: 您为RTTS数据集准备好的、包含所有图片路径和标签的总清单文件
full_rtts_list_file = './data/realworldtestfoggy/rtts.txt' 

# 输出: 划分后的三个清单文件
train_split_file = './data/rtts/rtts_train.txt'
val_split_file = './data/rtts/rtts_val.txt'
test_split_file = './data/rtts/rtts_test.txt'
# --- 配置结束 ---

# --- 2. 设置参数 ---
# 验证集和测试集各占15%，剩下70%是训练集
val_ratio = 0.1
test_ratio = 0.1
random_seed = 42 # 固定随机种子，保证划分结果可复现
# --------------------

def split_dataset():
    # 确保文件夹存在
    os.makedirs(os.path.dirname(train_split_file), exist_ok=True)

    print(f"正在从 {full_rtts_list_file} 读取数据...")
    with open(full_rtts_list_file, 'r') as f:
        lines = [line for line in f if line.strip()]
    
    if not lines:
        print("错误：文件为空或无法读取内容。")
        return

    # 设置随机种子并打乱列表顺序
    random.seed(random_seed)
    random.shuffle(lines)
    
    # 计算切分点
    total_size = len(lines)
    val_size = int(total_size * val_ratio)
    test_size = int(total_size * test_ratio)
    
    val_split_point = val_size
    test_split_point = val_size + test_size
    
    # 切分列表
    val_lines = lines[:val_split_point]
    test_lines = lines[val_split_point:test_split_point]
    train_lines = lines[test_split_point:]
    
    # 将划分结果写入新文件
    with open(train_split_file, 'w') as f:
        f.writelines(train_lines)
    with open(val_split_file, 'w') as f:
        f.writelines(val_lines)
    with open(test_split_file, 'w') as f:
        f.writelines(test_lines)
        
    print("\nRTTS 数据集划分完成！")
    print(f" -> 训练集大小: {len(train_lines)} -> 已保存到 {train_split_file}")
    print(f" -> 验证集大小: {len(val_lines)} -> 已保存到 {val_split_file}")
    print(f" -> 测试集大小: {len(test_lines)} -> 已保存到 {test_split_file}")

if __name__ == '__main__':
    split_dataset()