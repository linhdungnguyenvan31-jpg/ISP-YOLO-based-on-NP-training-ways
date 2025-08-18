# import numpy as np
# import os
# import cv2
# import math
# #from numba import jit
# import random
# from tqdm import tqdm
# import multiprocessing # 1. 导入多处理库

# # ===================================================================
# #  The functions load_annotations and parse_annotation remain exactly the same.
# #  They are already well-structured for parallel processing.
# # ===================================================================

# # 作用：从清单文件中读取所有需要处理的图片信息
# # only use the image including the labeled instance objects for training
# def load_annotations(annot_path):
#     print(f"正在读取清单文件: {annot_path}")
#     with open(annot_path, 'r') as f:
#         txt = f.readlines()
#         # --- 核心修改在这里 ---
#         # 只过滤掉完全是空白的行，从而保留所有正样本和负样本
#         annotations = [line.strip() for line in txt if line.strip()]
#     return annotations

# # 作用：为单张图片生成10张雾图
# def parse_annotation(annotation):
#     image_path_for_error_log = ""
#     try:
#         line = annotation.split()
#         image_path = line[0].replace('\\', '/')
#         image_path_for_error_log = image_path

#         # --- 核心改进 1: 智能跳过 ---
#         base_name = os.path.basename(image_path)
#         image_name_only, image_ext = os.path.splitext(base_name)
#         save_path_train = './data/voc_foggy/test/JPEGImages/' #train1表示的是正样本
        
#         # 快速检查，如果最后一个版本的雾图已存在，就认为已完成并跳过
#         last_foggy_file = os.path.join(save_path_train, f"{image_name_only}_0.14{image_ext}")
#         if os.path.exists(last_foggy_file):
#             return "skipped"

#         os.makedirs(save_path_train, exist_ok=True)
        
#         image = cv2.imread(image_path)
#         if image is None:
#             # print(f"警告：无法读取原始图片 {image_path}，已跳过。")
#             return "read_error"

#         for i in range(10):
#             # @jit() # 我们暂时禁用不稳定的Numba JIT
#             def AddHaz_loop(img_f, center, size, beta, A):
#                 (row, col, chs) = img_f.shape
#                 for j in range(row):
#                     for l in range(col):
#                         d = -0.04 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
#                         td = math.exp(-beta * d)
#                         img_f[j][l][:] = img_f[j][l][:] * td + A * (1 - td)
#                 return img_f

#             # 使用 float32 来节省内存
#             img_f = image.astype(np.float32) / 255.0
            
#             (row, col, chs) = image.shape
#             A = 0.5
#             beta = 0.01 * i + 0.05
#             size = math.sqrt(max(row, col))
#             center = (row // 2, col // 2)
#             foggy_image = AddHaz_loop(img_f, center, size, beta, A)
#             img_f = np.clip(foggy_image * 255, 0, 255)
#             img_f = img_f.astype(np.uint8)
            
#             new_filename = f"{image_name_only}_{beta:.2f}{image_ext}"
#             final_save_path = os.path.join(save_path_train, new_filename)
            
#             cv2.imwrite(final_save_path, img_f)
        
#         return "processed"

#     except Exception as e:
#         # --- 核心改进 2: 强大的容错性 ---
#         # print(f"处理 {image_path_for_error_log} 时发生严重错误: {e}，已跳过。")
#         return "process_error"

# # ===================================================================
# #  The main block is modified to use multiprocessing
# # ===================================================================

# # 主程序入口
# if __name__ == '__main__':
#     annotations = load_annotations('./data/dataset_fog/voc_norm_test.txt')
#     num_images = len(annotations)
    
#     if num_images > 0:
#         print(f"--- 清单中共有 {num_images} 张图片待处理 ---")
        
#         # 2. 智能选择核心数：使用一半的CPU核心，在速度和稳定性间取得最佳平衡
#         #    max(1, ...) 确保即使在单核机器上也能运行
#         num_cores_to_use = max(1, multiprocessing.cpu_count() // 2) 
#         print(f"--- 将使用 {num_cores_to_use} 个CPU核心进行并行处理 ---")

#         # 3. 创建并使用进程池
#         with multiprocessing.Pool(processes=num_cores_to_use) as pool:
#             # 4. 使用 pool.imap_unordered 来分发任务，并用tqdm显示进度条
#             # imap_unordered 性能更好，因为它不会等待结果的顺序
#             results = list(tqdm(pool.imap_unordered(parse_annotation, annotations), total=num_images))
            
#             # 5. 统计最终结果 (与单线程版完全相同)
#             skipped_count = results.count("skipped")
#             processed_count = results.count("processed")
#             error_count = len(results) - skipped_count - processed_count
            
#             print("\n--- 所有任务处理完毕！ ---")
#             print(f"--- 结果统计 ---")
#             print(f"已跳过 (之前已完成): {skipped_count} 张原始图片")
#             print(f"本次新处理: {processed_count} 张原始图片")
#             print(f"处理失败/跳过: {error_count} 张原始图片")


import os
import cv2
import numpy as np
import math
import random
from tqdm import tqdm
import multiprocessing
import argparse # 引入argparse，让脚本更灵活

# ===================================================================
# 辅助函数 (保持不变)
# ===================================================================

def load_annotations(annot_path):
    print(f"正在读取清单文件: {annot_path}")
    with open(annot_path, 'r') as f:
        txt = f.readlines()
        annotations = [line.strip() for line in txt if line.strip()]
    return annotations

def AddHaz_loop(img_f, center, size, beta, A):
    (row, col, chs) = img_f.shape
    for j in range(row):
        for l in range(col):
            d = -0.04 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
            td = math.exp(-beta * d)
            img_f[j][l][:] = img_f[j][l][:] * td + A * (1 - td)
    return img_f

# ===================================================================
# 核心工作函数 (已升级)
# ===================================================================

# 修正后的核心工作函数
def parse_annotation_for_test(args_tuple):
    """
    为单张测试图片生成一张随机等级的雾图，并返回新的清单行。
    """
    # 从元组中解包参数
    annotation, save_dir = args_tuple
    
    # 预先定义一个变量，以便在except中也能使用
    original_image_path = annotation.split()[0] if annotation.split() else "未知图片"

    try:
        # --- 核心修改：将 line 改为 annotation ---
        parts = annotation.strip().split()
        # ------------------------------------
        
        original_image_path = parts[0].replace('\\', '/')
        annotations_str = " ".join(parts[1:])

        base_name = os.path.basename(original_image_path)
        image_name_only, image_ext = os.path.splitext(base_name)
        
        # 随机选择一个雾天等级
        i = random.randint(0, 9)
        beta = 0.01 * i + 0.05
        
        new_filename = f"{image_name_only}_fog_beta_{beta:.2f}{image_ext}"
        final_save_path = os.path.join(save_dir, new_filename)
        
        # 智能跳过逻辑
        if os.path.exists(final_save_path):
            new_image_path = final_save_path.replace('\\', '/')
            return f"{new_image_path} {annotations_str}"

        os.makedirs(save_dir, exist_ok=True)
        
        original_image = cv2.imread(original_image_path)
        if original_image is None:
            print(f"警告: 无法读取图片 {original_image_path}")
            return None # 返回None表示失败

        # 不再使用循环
        img_f = original_image.astype(np.float32) / 255.0
        (row, col, chs) = original_image.shape
        A = 0.5; size = math.sqrt(max(row, col)); center = (row // 2, col // 2)
        foggy_image = AddHaz_loop(img_f, center, size, beta, A)
        img_f = np.clip(foggy_image * 255, 0, 255).astype(np.uint8)
        
        cv2.imwrite(final_save_path, img_f)
        
        # 返回新生成的行
        new_image_path = final_save_path.replace('\\', '/')
        return f"{new_image_path} {annotations_str}"

    except Exception as e:
        # 增加打印错误信息，方便未来调试
        print(f"处理 '{original_image_path}' 时发生错误: {e}")
        return None # 发生任何错误都返回None

# ===================================================================
# 主程序入口 (已升级)
# ===================================================================

if __name__ == '__main__':
    # 使用 argparse 让脚本更灵活
    parser = argparse.ArgumentParser(description="为测试集生成随机单等级雾图，并创建对应的清单文件。")
    parser.add_argument('--input_list', default='./data/dataset_fog/voc_norm_test.txt', help="输入的清晰测试图片清单文件路径")
    parser.add_argument('--save_dir', default='./data/voc_foggy/JPEGImages/', help="保存生成的雾天测试图片的文件夹路径")
    parser.add_argument('--output_list_file', default='./data/dataset_fog/voc_foggy_text.txt', help="输出的、包含雾图路径的新测试清单文件")
    args = parser.parse_args()

    lines = load_annotations(args.input_list)
    num_images = len(lines)
    
    if num_images > 0:
        print(f"--- 将为 {num_images} 张测试图片生成随机雾图 ---")
        
        num_cores_to_use = max(1, multiprocessing.cpu_count() // 2)
        print(f"--- 将使用 {num_cores_to_use} 个CPU核心进行并行处理 ---")

        tasks = [(line, args.save_dir) for line in lines]
        
        final_lines_to_write = []
        with multiprocessing.Pool(processes=num_cores_to_use) as pool:
            # imap_unordered 会保持任务分发的顺序，但结果返回的顺序是随机的，效率更高
            for result_line in tqdm(pool.imap_unordered(parse_annotation_for_test, tasks), total=num_images):
                if result_line: # 只收集有效的（非None）结果
                    final_lines_to_write.append(result_line)
        
        print(f"\n--- 增强处理完成，共生成 {len(final_lines_to_write)} 条新的测试数据 ---")
        print(f"--- 正在写入总列表文件到: {args.output_list_file} ---")
        
        # 一次性写入最终的清单文件
        with open(args.output_list_file, 'w') as f:
            for line in final_lines_to_write:
                f.write(line.strip() + '\n')
                
        print("--- 所有任务处理完毕！ ---")            
            
            
# import os
# import cv2
# import numpy as np
# import random
# from tqdm import tqdm
# import multiprocessing
# import argparse
# import albumentations as A

# # ===================================================================
# # 辅助函数：定义我们的数据增强“管道”
# # ===================================================================
# def get_transforms():
#     """定义您想要应用的所有数据增强操作。"""
#     transform = A.Compose([
#         #首先进行一些通用的操作
#         A.HorizontalFlip(p=0.5),
#         A.RandomBrightnessContrast(p=0.2),
#         # --- 核心：使用 A.OneOf 来随机选择一种天气效果 ---
#         A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.7, alpha_coef=0.1, p=0.8),
#         #A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, blur_value=7, brightness_coefficient=0.8, p=1.0),
#         #A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, brightness_coeff=2.5, p=1.0)
#         # p=1 意味着有100%的概率，会从上面三种天气中随机选择一种来应用到图片上
#     ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
#     # format='pascal_voc' 对应 [xmin, ymin, xmax, ymax] 格式
#     return transform

# # ===================================================================
# # 核心工作函数 (已重构，现在返回结果而不是写入文件)
# # ===================================================================
# def process_single_line(args_tuple):
#     """
#     处理清单文件中的单行数据，应用增强，保存增强后的图片，
#     并返回新图片路径和其对应的标注字符串。
#     """
#     line, save_dir_images, aug_idx = args_tuple
    
#     try:
#         parts = line.strip().split()
#         image_path = parts[0].replace('\\', '/')
        
#         bboxes = []
#         class_labels = []
#         for bbox_str in parts[1:]:
#             coords_and_id = bbox_str.split(',')
#             coords = [float(c) for c in coords_and_id[:4]]
#             class_id = int(coords_and_id[4])
#             xmin, ymin, xmax, ymax = coords
#             bboxes.append([xmin, ymin, xmax, ymax])
#             class_labels.append(class_id)

#         image = cv2.imread(image_path)
#         if image is None: return None
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
#         transforms = get_transforms()
#         transformed = transforms(image=image, bboxes=bboxes, class_labels=class_labels)
        
#         transformed_image = transformed['image']
#         transformed_bboxes = transformed['bboxes']
        
#         # 如果增强后所有标注框都被裁掉了，则跳过这张图片
#         if not transformed_bboxes:
#             return None
            
#         transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)

#         # --- 保存增强后的图片 ---
#         base_name = os.path.basename(image_path)
#         image_name_only, image_ext = os.path.splitext(base_name)
#         new_base_name = f"{image_name_only}_aug_{aug_idx}"
#         final_image_path = os.path.join(save_dir_images, f"{new_base_name}{image_ext}").replace('\\', '/')
#         cv2.imwrite(final_image_path, transformed_image)
        
#         # --- 构建非归一化的标注字符串 ---
#         annotations_for_this_image = []
#         for bbox, class_id in zip(transformed_bboxes, transformed['class_labels']):
#             # Albumentations 返回的已经是变换后的原始像素坐标
#             xmin, ymin, xmax, ymax = bbox
            
#             # 直接使用整数坐标，不进行归一化
#             ann_str = f"{int(xmin)},{int(ymin)},{int(xmax)},{int(ymax)},{class_id}"
#             annotations_for_this_image.append(ann_str)
            
#         # 返回要写入总列表文件的完整行
#         return f"{final_image_path} {' '.join(annotations_for_this_image)}"

#     except Exception as e:
#         return None

# # ===================================================================
# # 主程序入口 (已重构)
# # ===================================================================
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="使用Albumentations为数据集并行进行高级数据增强，并生成IA-YOLO格式的列表文件。")
#     parser.add_argument('--annot_path', type=str, default = "./data/test.txt", help="包含图片路径和多目标标注的清单文件路径")
#     parser.add_argument('--save_dir', type=str, default = "./data/savetest", help="保存增强后数据的根目录")
#     parser.add_argument('--num_augmentations', type=int, default=5, help="为每张原始图片生成多少张增强图片")
#     args = parser.parse_args()

#     with open(args.annot_path, 'r') as f:
#         lines = f.readlines()
    
#     # 构建保存增强后图片的目录
#     save_dir_images = os.path.join(args.save_dir, 'augmented_images')
#     os.makedirs(save_dir_images, exist_ok=True)
    
#     # 构建最终输出的总列表文件路径
#     output_list_file = os.path.join(args.save_dir, 'train_augmented.txt')

#     # 为每张图片创建N个增强任务
#     tasks = []
#     for i in range(args.num_augmentations):
#         for line in lines:
#             tasks.append((line, save_dir_images, i))

#     print(f"--- 共 {len(lines)} 张原始图片，将生成 {len(tasks)} 张增强图片 ---")
    
#     # --- 核心修改：先并行处理，再统一写入 ---
#     # 1. 使用多进程池执行所有增强任务，并将返回的“行字符串”收集起来
#     num_cores = max(1, multiprocessing.cpu_count() // 2)
#     results = []
#     with multiprocessing.Pool(processes=num_cores) as pool:
#         # tqdm显示总任务进度
#         for result in tqdm(pool.imap_unordered(process_single_line, tasks), total=len(tasks)):
#             # 只收集有效的（非None）结果
#             if result:
#                 results.append(result)

#     # 2. 所有并行任务完成后，一次性写入总列表文件
#     print(f"\n--- 增强处理完成，共生成 {len(results)} 张有效增强图片 ---")
#     print(f"--- 正在写入总列表文件到: {output_list_file} ---")
#     with open(output_list_file, 'w') as f:
#         for line in results:
#             f.write(line + '\n')
            
#     print("--- 所有任务处理完毕！ ---")

# import os
# import cv2
# import numpy as np
# import math
# import random
# from tqdm import tqdm
# import multiprocessing
# import argparse

# # ===================================================================
# # 辅助函数部分 (Helper Functions)
# # ===================================================================

# def load_annotations(annot_path):
#     """从清单文件中读取所有需要处理的图片信息"""
#     print(f"正在读取清单文件: {annot_path}")
#     with open(annot_path, 'r') as f:
#         txt = f.readlines()
#         annotations = [line.strip() for line in txt if len(line.strip().split()) > 1]
#     return annotations

# def AddHaz_loop(img_f, center, size, beta, A):
#     """原始增雾函数 (像素级循环)"""
#     (row, col, chs) = img_f.shape
#     for j in range(row):
#         for l in range(col):
#             d = -0.04 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
#             td = math.exp(-beta * d)
#             img_f[j][l][:] = img_f[j][l][:] * td + A * (1 - td)
#     return img_f

# def add_rain(image, slant, drop_length, drop_width, drop_color, blur_value, brightness_coefficient, rain_drops):
#     """高效的人工增雨函数 (矢量化操作)"""
#     rainy_image = np.copy(image)
#     height, width, _ = rainy_image.shape
#     rain_layer = np.zeros((height, width), dtype=np.uint8)
#     for _ in range(rain_drops):
#         x1 = random.randint(-width // 4, width + width // 4)
#         y1 = random.randint(-height // 4, height + height // 4)
#         x2 = x1 + slant
#         y2 = y1 + drop_length
#         cv2.line(rain_layer, (x1, y1), (x2, y2), 255, drop_width)
#     rain_layer = cv2.blur(rain_layer, (blur_value, blur_value))
#     rain_layer_color = np.zeros_like(rainy_image)
#     for i in range(3):
#         rain_layer_color[:, :, i] = rain_layer * (drop_color[i] / 255.0)
#     rainy_image = cv2.convertScaleAbs(rainy_image, alpha=brightness_coefficient, beta=0)
#     rainy_image = cv2.add(rainy_image, rain_layer_color)
#     return rainy_image

# # --- 新增：人工增雪函数 ---
# def add_snow(image, snow_flakes=5000, flake_size_range=(1, 4), brightness_coeff=1.2):
#     """
#     为一个图像添加下雪效果。
#     """
#     snowy_image = np.copy(image)
#     height, width, _ = snowy_image.shape
    
#     # 稍微提高画面亮度，模拟雪地反光
#     snowy_image = cv2.convertScaleAbs(snowy_image, alpha=brightness_coeff, beta=10)
    
#     # 创建雪花层
#     snow_layer = np.zeros((height, width, 3), dtype=np.uint8)
    
#     for _ in range(snow_flakes):
#         x = random.randint(0, width)
#         y = random.randint(0, height)
#         size = random.randint(flake_size_range[0], flake_size_range[1])
#         color = random.randint(180, 255) # 灰白色雪花
#         cv2.circle(snow_layer, (x, y), size, (color, color, color), -1)

#     # 对雪花层进行模糊，使其更自然
#     snow_layer = cv2.GaussianBlur(snow_layer, (3, 3), 0)

#     # 将雪花层和图像融合
#     snowy_image = cv2.addWeighted(snowy_image, 0.8, snow_layer, 0.2, 0)
    
#     return snowy_image

# # ===================================================================
# # 核心工作函数 (Core Worker Function)
# # ===================================================================
# def process_and_augment_image(args_tuple):
#     """为单张图片生成指定的增强效果，并返回新生成的训练列表行列表。"""
#     line, aug_type, save_dir = args_tuple
    
#     try:
#         parts = line.strip().split()
#         original_image_path = parts[0].replace('\\', '/')
#         annotations_str = " ".join(parts[1:])

#         original_image = cv2.imread(original_image_path)
#         if original_image is None: return []

#         base_name = os.path.basename(original_image_path)
#         image_name_only, image_ext = os.path.splitext(base_name)
#         os.makedirs(save_dir, exist_ok=True)
        
#         new_lines_for_training_list = []

#         if aug_type == 'fog':
#             for i in range(10):
#                 beta = 0.01 * i + 0.05
#                 # ... (省略增雾的具体实现，与之前版本相同)
#                 new_filename = f"{image_name_only}_fog_beta_{beta:.2f}{image_ext}"
#                 final_save_path = os.path.join(save_dir, new_filename)
#                 if os.path.exists(final_save_path): continue #断点续传功能 
#                 img_f = original_image.astype(np.float32) / 255.0
#                 (row, col, chs) = original_image.shape
#                 A = 0.5; size = math.sqrt(max(row, col)); center = (row // 2, col // 2)
#                 foggy_image = AddHaz_loop(img_f, center, size, beta, A)
#                 img_f = np.clip(foggy_image * 255, 0, 255).astype(np.uint8)
#                 cv2.imwrite(final_save_path, img_f)
#                 new_image_path = final_save_path.replace('\\', '/')
#                 new_lines_for_training_list.append(f"{new_image_path} {annotations_str}")

#         elif aug_type == 'rain':
#             for level in range(1, 6):
#                 new_filename = f"{image_name_only}_rain_level_{level}{image_ext}"
#                 # ... (省略增雨的具体实现，与之前版本相同)
#                 final_save_path = os.path.join(save_dir, new_filename)
#                 if os.path.exists(final_save_path): continue
#                 rain_drops = 500 + level * 250; brightness = 0.95 - level * 0.06
#                 rainy_image = add_rain(image=original_image, rain_drops=rain_drops, brightness_coefficient=brightness, slant=random.randint(-15, -5), drop_length=25, drop_width=1, drop_color=(200, 200, 200), blur_value=5)
#                 cv2.imwrite(final_save_path, rainy_image)
#                 new_image_path = final_save_path.replace('\\', '/')
#                 new_lines_for_training_list.append(f"{new_image_path} {annotations_str}")
        
#         # --- 新增：增雪逻辑 ---
#         elif aug_type == 'snow':
#             for level in range(1, 6): # 生成5个等级的雪
#                 new_filename = f"{image_name_only}_snow_level_{level}{image_ext}"
#                 final_save_path = os.path.join(save_dir, new_filename)
                
#                 if os.path.exists(final_save_path): continue

#                 snow_flakes = 1000 + level * 2000 # 雪花越来越多
#                 brightness = 1.1 + level * 0.05 # 天色越来越亮
#                 snowy_image = add_snow(image=original_image, snow_flakes=snow_flakes, brightness_coeff=brightness)
#                 cv2.imwrite(final_save_path, snowy_image)
                
#                 new_image_path = final_save_path.replace('\\', '/')
#                 new_lines_for_training_list.append(f"{new_image_path} {annotations_str}")

#         return new_lines_for_training_list
#     except Exception as e:
#         return []

# # ===================================================================
# # 主程序入口 (Main Block)
# # ===================================================================
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="为数据集并行添加“雾”、“雨”、“雪”等效果，并生成新的训练清单。")
    
#     parser.add_argument('--annot_path', type=str, default='./data/coco_fog/train_5_classes.txt', help="输入的训练清单文件路径")
#     parser.add_argument('--aug_type', type=str, default='fog', choices=['fog', 'rain', 'snow'], help="选择增强类型: 'fog', 'rain' 或 'snow'")
#     parser.add_argument('--save_dir', type=str, default = ".data/augumented_foggyimages/JPEGImages",help = "存储图片路径")
#     parser.add_argument('--num_cores', type=int, default=max(1, multiprocessing.cpu_count() // 2), help="使用的CPU核心数")
#     args = parser.parse_args()
    
#     lines = load_annotations(args.annot_path)
#     random.shuffle(lines)
#     num_images = len(lines)
#     p25 = int(num_images * 0.25)
#     p50 = int(num_images * 0.50)
#     p75 = int(num_images * 0.75)
    
#     fog_imags = num_images[:p25]
#     rain_imags = num_images[p25:p50]
#     snow_imags = num_images[p50:p75]
#     clear_imags = num_images[p75:]
    
#     if num_images > 0:
#         final_image_save_dir = os.path.join(args.save_dir, f'coco_{args.aug_type}')
#         output_list_file = os.path.join(args.save_dir, f'train_{args.aug_type}_list.txt')
        
#         print(f"--- 增强类型: {args.aug_type} ---")
#         print(f"--- 增强后的图片将保存到: {final_image_save_dir} ---")
#         print(f"--- 新的训练清单将保存到: {output_list_file} ---")
        
        
#         tasks = [(line, args.aug_type, final_image_save_dir) for line in lines]
        
#         all_results = []
#         with multiprocessing.Pool(processes=args.num_cores) as pool:
#             for result_list in tqdm(pool.imap_unordered(process_and_augment_image, tasks), total=num_images):
#                 if result_list:
#                     all_results.append(result_list)
        
#         final_lines_to_write = []
#         for sublist in all_results:
#             final_lines_to_write.extend(sublist)
            
#         print(f"\n--- 增强处理完成，共生成 {len(final_lines_to_write)} 条新的训练数据 ---")
#         print(f"--- 正在写入总列表文件到: {output_list_file} ---")
#         # print(f"已跳过 (之前已完成): {skipped_count} 张原始图片")
#         # print(f"本次新处理: {processed_count} 张原始图片")
#         # print(f"处理失败/跳过: {error_count} 张原始图片")
#         with open(output_list_file, 'w') as f:
#             for line in final_lines_to_write:
#                 f.write(line + '\n')
                
#         print("--- 所有任务处理完毕！ ---")