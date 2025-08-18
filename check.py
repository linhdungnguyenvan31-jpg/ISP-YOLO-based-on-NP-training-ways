from pycocotools.coco import COCO

# 替换成您的真实路径
annotation_file = './data/Coco2017/annotations/instances_train2017.json'

try:
    print(f"正在加载文件: {annotation_file}")
    coco = COCO(annotation_file)
    
    # 获取图片总数
    all_img_ids = coco.getImgIds()
    
    # 【新增】获取所有标注的总数
    all_ann_ids = coco.getAnnIds()
    
    print("\n--- 最终诊断结果 ---")
    print(f"图片总数: {len(all_img_ids)}")
    print(f"标注总数 (Annotations): {len(all_ann_ids)}")
    print("--------------------")

except Exception as e:
    print(f"加载文件时出错: {e}")