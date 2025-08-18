import os
import argparse
from tqdm import tqdm

def analyze_annotation_file(file_path):
    """
    分析一个标注清单文件，统计正、负样本和总数。
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 '{file_path}'")
        return

    # 初始化计数器
    total_lines = 0
    positive_samples = 0 # 有物体的图片 (正样本)
    negative_samples = 0 # 无物体的图片 (负样本)

    print("正在打开文件并逐行分析，请稍候...")
    
    # 使用 with open 安全地打开文件
    with open(file_path, 'r') as f:
        # readlines() 对于中等大小的文件是OK的
        lines = f.readlines()
        
        # 使用 tqdm 来显示处理进度
        for line in tqdm(lines, desc="Analyzing lines"):
            # 清理行首尾的空白字符
            clean_line = line.strip()
            
            # 如果是空行，则跳过
            if not clean_line:
                continue
            
            # 统计有效行（即图片总数）
            total_lines += 1
            
            # 按空格分割
            parts = clean_line.split()
            
            # 核心判断逻辑：
            # 如果分割后的部分大于1个，说明除了图片路径外，至少还有1个标注
            if len(parts) > 1:
                positive_samples += 1
            else:
                negative_samples += 1

    # 打印最终的统计报告
    print("\n" + "="*30)
    print("--- 数据集分析报告 ---")
    print(f"文件路径: {file_path}")
    print("-"*30)
    print(f"图片总数 (Total Images): {total_lines}")
    print(f"有物体的图片 (Positive Samples): {positive_samples}")
    print(f"无物体的图片 (Negative Samples): {negative_samples}")
    print("="*30)

# 主程序入口
if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="分析目标检测清单文件，统计正、负样本数量。")
    # 添加 --file_path 参数，让用户从命令行指定要分析的文件
    parser.add_argument("--file_path", required=True, help="需要分析的清单文件路径 (例如 a.txt)")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用分析函数
    analyze_annotation_file(args.file_path)