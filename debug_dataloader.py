import os
import cv2
import numpy as np

# --- 从您的项目中导入Dataset类和配置 ---
# 确保这个调试脚本和 train.py, core/ 等在同一个目录下
print("正在导入项目模块...")
try:
    from core.dataset import Dataset
    from core.config import cfg
    print("项目模块导入成功！")
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保此脚本位于您 IA_yolo 项目的主目录下。")
    exit()

# --- 配置 ---
OUTPUT_DIR = './debug_dataloader_output'
NUM_SAMPLES_TO_CHECK = 6 # 从一个批次中检查几张图片
# ------------

def decode_and_draw_boxes(image, label_s, label_m, label_l):
    """
    一个辅助函数，用于从YOLO的标签格式中解码出边界框并画在图上。
    """
    # 遍历小、中、大三个尺度的标签
    for label in [label_s, label_m, label_l]:
        # 寻找有物体的位置 (置信度 > 0)
        true_box_coords = label[..., 0:4][label[..., 4] == 1]
        
        for box in true_box_coords:
            # 将 [x_center, y_center, w, h] 转换为 [xmin, ymin, xmax, ymax]
            x_center, y_center, w, h = box
            xmin = int(x_center - w / 2)
            ymin = int(y_center - h / 2)
            xmax = int(x_center + w / 2)
            ymax = int(y_center + h / 2)
            # 用绿色(BGR)画出解码后的真实框
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    return image

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\n--- 正在初始化训练集 Dataset('train')... ---")
    # 初始化您的数据集类 (用于训练)
    train_dataset = Dataset('train') 

    print("--- 正在从数据加载器中取出一个批次的数据... ---")
    try:
        # 从加载器中取出一个批次的数据
        batch_data = next(train_dataset)
    except StopIteration:
        print("错误：数据加载器为空。")
        return

    batch_image, batch_label_sbbox, batch_label_mbbox, batch_label_lbbox, \
    _, _, _, batch_clean_image = batch_data

    print(f"--- 成功获取一批数据，将可视化其中的 {NUM_SAMPLES_TO_CHECK} 张图片 ---")

    # 遍历这个批次里的每一张图片
    for i in range(min(train_dataset.batch_size, NUM_SAMPLES_TO_CHECK)):
        # --- 处理 foggy_image (模型的输入) ---
        foggy_tensor = batch_image[i]
        foggy_img_np = (foggy_tensor * 255).astype(np.uint8) # 反归一化
        foggy_img_bgr = cv2.cvtColor(foggy_img_np, cv2.COLOR_RGB2BGR) # 转回BGR
        
        # --- 处理 clean_image (标准答案) ---
        clean_tensor = batch_clean_image[i]
        clean_img_np = (clean_tensor * 255).astype(np.uint8)
        clean_img_bgr = cv2.cvtColor(clean_img_np, cv2.COLOR_RGB2BGR)
        
        # --- 为两张图片都画上标签框，以供对比 ---
        foggy_with_boxes = decode_and_draw_boxes(foggy_img_bgr, batch_label_sbbox[i], batch_label_mbbox[i], batch_label_lbbox[i])
        clean_with_boxes = decode_and_draw_boxes(clean_img_bgr, batch_label_sbbox[i], batch_label_mbbox[i], batch_label_lbbox[i])
        
        # --- 保存成对的图片 ---
        save_path_foggy = os.path.join(OUTPUT_DIR, f'sample_{i}_A_ModelInput.png')
        save_path_clean = os.path.join(OUTPUT_DIR, f'sample_{i}_B_GroundTruth.png')
        cv2.imwrite(save_path_foggy, foggy_with_boxes)
        cv2.imwrite(save_path_clean, clean_with_boxes)

    print(f"\n--- 检查完毕！请打开文件夹 '{OUTPUT_DIR}' 并对比成对的图片。 ---")

if __name__ == '__main__':
    main()