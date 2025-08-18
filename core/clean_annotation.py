import os
from tqdm import tqdm

# --- 配置 ---
#original_annotation_file = './data/dataset_fog/voc_norm_train.txt' # 原始清单
original_annotation_file = './data/dataset_fog/voc_norm_test.txt' # 原始清单
cleaned_annotation_file = './data/dataset_fog/test_clean.txt'   # 新的、干净的清单
#cleaned_annotation_file = './data/dataset_fog/train_clean.txt'   # 新的、干净的清单
foggy_image_dir = './data/voc_foggy/test/JPEGImages/'         # 你存放雾图的文件夹
#foggy_image_dir = './data/voc_foggy/train/JPEGImages/'         # 你存放雾图的文件夹

# --- 主程序 ---
print("开始清理标注清单文件...")

with open(original_annotation_file, 'r') as f_in, open(cleaned_annotation_file, 'w') as f_out:
    lines = f_in.readlines()
    for line in tqdm(lines):
        parts = line.strip().split()
        if not parts:
            continue
            
        image_path = parts[0].replace('\\', '/')
        base_name = os.path.basename(image_path)
        image_name_only, image_ext = os.path.splitext(base_name)
        
        # 检查第一张生成的雾图是否存在 (beta=0.05)
        # 如果存在，我们才认为这张原始图片是“好”的
        expected_foggy_file = os.path.join(foggy_image_dir, f"{image_name_only}_0.05{image_ext}")
        
        if os.path.exists(expected_foggy_file):
            f_out.write(line) # 把这行“好”数据写入到新文件中

print(f"清理完成！新的清单文件已保存到: {cleaned_annotation_file}")