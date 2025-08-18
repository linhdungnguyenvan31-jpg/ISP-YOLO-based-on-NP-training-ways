import os
import shutil
from tqdm import tqdm
import argparse

def organize_rtts_by_lists(base_rtts_dir, train_list_path, val_list_path, test_list_path, output_dir):
    """
    根据训练、验证、测试清单文件，将RTTS数据集的图片整理到新的目录结构中。
    """
    print("--- 开始整理RTTS数据集 ---")
    
    # 1. 定义新的目标文件夹路径
    train_img_dir = os.path.join(output_dir, 'train', 'JPEGImages')
    val_img_dir = os.path.join(output_dir, 'val', 'JPEGImages')
    test_img_dir = os.path.join(output_dir, 'test', 'JPEGImages')
    
    # 2. 创建这些新的文件夹
    print("正在创建目标文件夹...")
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)
    print("目标文件夹创建成功！")

    # 3. 定义一个辅助函数来处理单个清单文件
    def process_list(list_path, destination_dir):
        print(f"\n--- 正在处理清单: {list_path} ---")
        if not os.path.exists(list_path):
            print(f"警告：找不到清单文件 {list_path}，已跳过。")
            return 0

        with open(list_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        copied_count = 0
        for line in tqdm(lines, desc=f"复制到 {os.path.basename(destination_dir)}"):
            try:
                # 假设清单文件中的路径是相对于某个基础目录的
                # 我们先提取出文件名
                file_name = os.path.basename(line.split()[0]) #提取出每张图片带后缀得文件名
                
                # 构建原始文件的完整路径
                source_path = os.path.join(base_rtts_dir, 'JPEGImages', file_name)  
                
                # 构建目标文件的完整路径
                destination_path = os.path.join(destination_dir, file_name)
                
                # 只有在源文件存在时才复制
                if os.path.exists(source_path):
                    shutil.copy(source_path, destination_path)
                    copied_count += 1
                else:
                    print(f"\n警告：在源目录中找不到图片 {source_path}，已跳过。")

            except Exception as e:
                print(f"\n处理行 '{line}' 时发生错误: {e}")
        
        return copied_count

    # 4. 依次处理训练、验证、测试集
    num_train = process_list(train_list_path, train_img_dir)
    num_val = process_list(val_list_path, val_img_dir)
    num_test = process_list(test_list_path, test_img_dir)
    
    print("\n--- 数据集整理完毕！ ---")
    print(f"成功复制了 {num_train} 张训练图片。")
    print(f"成功复制了 {num_val} 张验证图片。")
    print(f"成功复制了 {num_test} 张测试图片。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="根据清单文件，自动整理RTTS数据集的目录结构。")
    parser.add_argument('--base_dir', default = './data/RTTS/' , help="原始RTTS数据集的总文件夹路径 (应包含JPEGImages子文件夹)")
    parser.add_argument('--train_list', default = './data/rtts/rtts_train.txt', help="训练集清单文件 (.txt) 的路径")
    parser.add_argument('--val_list', default = './data/rtts/rtts_val.txt', help="验证集清单文件 (.txt) 的路径")
    parser.add_argument('--test_list', default ='./data/rtts/rtts_test.txt' , help="测试集清单文件 (.txt) 的路径")
    parser.add_argument('--output_dir', default = './data/rttsimages', help="整理后，新数据集的存放路径")
    
    args = parser.parse_args()
    
    organize_rtts_by_lists(args.base_dir, args.train_list, args.val_list, args.test_list, args.output_dir)