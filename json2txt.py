import os
from pycocotools.coco import COCO
from tqdm import tqdm

"""
最终修正版脚本：
加载COCO2017官方标注，通过手动合并类别的方式绕过pycocotools潜在的bug，
筛选出指定的5个类别，转换坐标格式，并生成一个单一的列表文件。
"""

# --- 配置部分 ---
# 请确保使用我们确认过的、完整的、新下载的标注文件的绝对路径
annotation_file ='./data/Coco2017/annotations/instances_val2017.json'

# COCO图片所在的目录 (请确保这是您存放图片的真实路径)
image_dir = './data/Coco2017/val2017'

# 最终输出的总列表文件名
output_list_file = './data/coco_fog/val_5_classes.txt'

# 目标类别列表 (已修正为官方名称 motorcycle)
TARGET_CLASSES = ['person', 'car', 'bus', 'bicycle', 'motorcycle']
# --- 修改结束 ---

def convert_coco_to_custom_format():
    # --- 1. 初始化和准备工作 ---
    print("正在加载COCO标注文件...")
    coco = COCO(annotation_file)
    print("标注文件加载并索引完毕！")

    class_mapping = {name: i for i, name in enumerate(TARGET_CLASSES)}
    target_coco_cat_ids = coco.getCatIds(catNms=TARGET_CLASSES)

    # ==================== 核心修正：手动构建图片ID的并集（Union）来绕过库的BUG ====================
    print("正在手动构建图片ID列表以绕过pycocotools的潜在BUG...")
    
    # 1. 为每个类别单独获取图片ID，并将它们变成集合(set)
    image_ids_per_category = []
    for cat_id in target_coco_cat_ids:
        ids = set(coco.getImgIds(catIds=[cat_id]))
        image_ids_per_category.append(ids)
        
    # 2. 使用 set.union() 安全地合并所有ID集合，得到并集
    union_img_ids = set.union(*image_ids_per_category)
    
    # 3. 转换为列表，准备处理
    img_ids = list(union_img_ids)
    
    print(f"修正后的筛选逻辑完成！共找到 {len(img_ids)} 张包含目标类别的图片。开始处理...")
    # ================================== 修正结束 ==================================

    # --- 2. 遍历图片并转换标注 ---
    with open(output_list_file, 'w', encoding='utf-8') as f_out:
        for img_id in tqdm(img_ids, desc="Processing Images"):
            img_info = coco.loadImgs(img_id)[0]
            file_name = img_info['file_name']
            image_path = os.path.join(image_dir, file_name).replace('\\', '/')
            
            ann_ids = coco.getAnnIds(imgIds=img_id, catIds=target_coco_cat_ids)
            annotations = coco.loadAnns(ann_ids)

            annotations_for_this_image = []
            for ann in annotations:
                coco_cat_id = ann['category_id']
                coco_cat_name = coco.loadCats(coco_cat_id)[0]['name']
                new_class_id = class_mapping[coco_cat_name]

                bbox = ann['bbox']
                x, y, w, h = bbox
                xmin = int(x)
                ymin = int(y)
                xmax = int(x + w)
                ymax = int(y + h)
                
                ann_str = f"{xmin},{ymin},{xmax},{ymax},{new_class_id}"
                annotations_for_this_image.append(ann_str)
            
            if annotations_for_this_image:
                line_to_write = f"{image_path} {' '.join(annotations_for_this_image)}\n"
                f_out.write(line_to_write)

    print(f"\n转换完成！结果已保存到: {output_list_file}")


if __name__ == '__main__':
    convert_coco_to_custom_format()
    