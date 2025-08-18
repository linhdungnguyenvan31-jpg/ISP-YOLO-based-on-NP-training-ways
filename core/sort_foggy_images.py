import os
import shutil
from tqdm import tqdm

# --- 配置 ---
# 原始的、包含所有雾图的大文件夹
source_foggy_dir = './data/voc_foggy/train/JPEGImages/'

# 新的、分拣后的训练集和验证集雾图文件夹
dest_train_dir = './data/voc_foggy/voc_foggy_split/train/JPEGImages/'
dest_val_dir = './data/voc_foggy/voc_foggy_split/val/JPEGImages/'

# 两个新的清单文件
train_list_file = './data/dataset_fog/train_final.txt'
val_list_file = './data/dataset_fog/val_final.txt'

# --- 主程序 ---

def sort_images(list_file, destination_dir):
    """根据清单文件，将雾图从源文件夹移动到目标文件夹"""
    print(f"\n正在处理清单: {list_file}")
    print(f"目标文件夹: {destination_dir}")

    if not os.path.exists(list_file):
        print(f"错误: 清单文件不存在 {list_file}")
        return

    os.makedirs(destination_dir, exist_ok=True)

    with open(list_file, 'r') as f:
        lines = f.readlines()

        for line in tqdm(lines, desc=f"正在分拣图片到 {destination_dir}"):
            parts = line.strip().split()
            if not parts:
                continue

            original_image_path = parts[0].replace('\\', '/')
            base_name = os.path.basename(original_image_path)
            image_name_only, image_ext = os.path.splitext(base_name)

            # 为一张原始图片，找到它对应的10张雾图并移动
            for i in range(10):
                beta = 0.01 * i + 0.05
                foggy_filename = f"{image_name_only}_{beta:.2f}{image_ext}"

                source_file = os.path.join(source_foggy_dir, foggy_filename)
                dest_file = os.path.join(destination_dir, foggy_filename)

                if os.path.exists(source_file):
                    shutil.move(source_file, dest_file)

if __name__ == '__main__':
    # 分拣训练集
    sort_images(train_list_file, dest_train_dir)
    # 分拣验证集
    sort_images(val_list_file, dest_val_dir)
    print("\n所有雾天图片已成功分拣到新的训练集和验证集文件夹")