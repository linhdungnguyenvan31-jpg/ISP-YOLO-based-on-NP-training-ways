# check_dataset.py
import core.dataset as dataset
from tqdm import tqdm
import time

print("--- 开始进行数据集加载健康体检 ---")
print("--- 正在初始化训练数据集...")

try:
    # 1. 创建一个数据集的实例，和 train.py 里做的一样
    train_dataset = dataset.Dataset('train')

    # 2. 尝试用一个 for 循环来遍历整个数据集
    # tqdm 会为我们显示一个进度条
    print(f"--- 数据集包含 {train_dataset.num_batchs} 个批次，现在开始遍历...")

    # 记录开始时间
    start_time = time.time()

    for i, batch_data in enumerate(tqdm(train_dataset)):
        # 我们不需要对数据做任何事，只要能成功加载出来就行
        # 每成功加载一个批次，就打印一次信息
        print(f"成功加载第 {i+1}/{train_dataset.num_batchs} 批次的数据！")

    end_time = time.time()
    print(f"\n--- ✅ 健康体检通过！成功遍历所有数据，总耗时: {end_time - start_time:.2f} 秒 ---")

except Exception as e:
    print(f"\n--- ❌ 健康体检失败！在加载过程中发生错误 ---")
    print(f"错误类型: {type(e).__name__}")
    print(f"错误信息: {e}")