from pycocotools.coco import COCO
import os
import random

# --- 准备工作 ---
annotation_file = './data/Coco2017/annotations/instances_train2017.json'
print("正在加载标注文件，请稍候...")
coco = COCO(annotation_file)
print("标注文件加载并索引完毕！\n")

# --- 核心步骤：按类别查找 ---

# 1. 获取所有类别的ID
# cat_ids = coco.getCatIds() # 这会返回所有类别ID的列表

# 2. 我们可以直接指定我们感兴趣的类别名称
#    例如: 'person', 'car', 'bus', 'motorcycle', 'bicycle'
category_name = 'car'
cat_ids = coco.getCatIds(catNms=[category_name])
print(f"找到了类别 '{category_name}' 的ID: {cat_ids}")

# 3. 获取所有包含这个类别的图片ID
#    这会返回一个包含成千上万个图片ID的列表
img_ids_with_person = coco.getImgIds(catIds=cat_ids)
print(f"数据集中共有 {len(img_ids_with_person)} 张图片包含 '{category_name}'。")

# 4. 从这个列表中随机选择一个ID，这个ID保证有标注！
if img_ids_with_person:
    random_img_id = random.choice(img_ids_with_person)
    print(f"\n我们随机选择了一张包含 '{category_name}' 的图片，ID为: {random_img_id}")

    # --- 现在用这个保证正确的ID来获取标注 ---
    ann_ids = coco.getAnnIds(imgIds=random_img_id)
    annotations = coco.loadAnns(ann_ids)

    print("\n--- 该图片的标注信息如下 ---")
    for ann in annotations:
        # 获取类别名称，让输出更易读
        category_id = ann['category_id']
        category_info = coco.loadCats(category_id)[0]
        class_name = category_info['name']
        
        print(f"类别: {class_name} (ID: {category_id})")
        print(f"边界框: {ann['bbox']}")
        print('---')
else:
    print(f"错误：在数据集中没有找到任何关于 '{category_name}' 的标注。")