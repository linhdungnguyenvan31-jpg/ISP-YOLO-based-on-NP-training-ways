import os
from pycocotools.coco import COCO

# --- 配置部分 ---
# 请确保这个路径是绝对路径，并且指向那个我们确认过是完整的、约556MB的标注文件
annotation_file = './data/Coco2017/annotations/instances_train2017.json'
# --- 修改结束 ---

def run_diagnostics():
    if not os.path.exists(annotation_file):
        print(f"[!!!] 致命错误：找不到标注文件: {annotation_file}")
        print("[!!!] 请确保上面的路径是正确的！")
        return

    print("正在加载COCO标注文件...")
    coco = COCO(annotation_file)
    print("加载完毕！")
    print("\n" + "="*50)
    print("--- 开始诊断筛选逻辑 ---")
    print("="*50)

    # --- 诊断1：逐个检查每个类别的图片数量 ---
    print("\n[诊断1] 检查单个类别...")
    classes_to_test = ['person', 'car', 'bus', 'bicycle', 'motorcycle', 'motorbike']
    cat_id_map = {}
    for name in classes_to_test:
        # 获取该类别在COCO中的ID
        cat_ids = coco.getCatIds(catNms=[name])
        if not cat_ids:
            print(f"  -> 类别 '{name}': 未在数据集中找到 (这对于 'motorbike' 是正常的)")
            continue
        
        # 存储下来方便后续使用
        cat_id_map[name] = cat_ids[0]
        
        # 获取包含该类别的图片ID列表
        img_ids = coco.getImgIds(catIds=cat_ids)
        print(f"  -> 仅包含 '{name}' (ID: {cat_ids[0]}) 的图片数量: {len(img_ids)}")

    print("\n" + "="*50)

    # --- 诊断2：检查“或”逻辑（包含任意一个） ---
    print("\n[诊断2] 检查“或”逻辑（Union / OR）...")
    TARGET_CLASSES_CORRECT = ['person', 'car', 'bus', 'bicycle', 'motorcycle']
    target_cat_ids = coco.getCatIds(catNms=TARGET_CLASSES_CORRECT)
    print(f"  -> 查询的5个正确类别的ID: {target_cat_ids}")
    
    # 正常情况下，这会返回包含这5个类别中【任意一个】的所有图片
    img_ids_union = coco.getImgIds(catIds=target_cat_ids)
    print(f"  -> 理论上包含这5个类别中【任意一个】的图片总数: {len(img_ids_union)}")
    
    print("\n" + "="*50)

    # --- 诊断3：检查“与”逻辑（必须同时包含） ---
    print("\n[诊断3] 检查“与”逻辑（Intersection / AND）...")
    print("  -> 这部分是为了验证一个猜想，计算可能需要一点时间...")
    
    # 获取每个类别的图片ID集合
    person_img_ids = set(coco.getImgIds(catIds=coco.getCatIds(catNms=['person'])))
    car_img_ids = set(coco.getImgIds(catIds=coco.getCatIds(catNms=['car'])))
    bus_img_ids = set(coco.getImgIds(catIds=coco.getCatIds(catNms=['bus'])))
    bicycle_img_ids = set(coco.getImgIds(catIds=coco.getCatIds(catNms=['bicycle'])))
    motorcycle_img_ids = set(coco.getImgIds(catIds=coco.getCatIds(catNms=['motorcycle'])))
    
    # 计算交集
    intersection_ids = person_img_ids.intersection(car_img_ids, bus_img_ids, bicycle_img_ids, motorcycle_img_ids)
    print(f"  -> **必须同时包含** 这5个类别的图片总数: {len(intersection_ids)}")
    
    print("\n" + "="*50)
    print("--- 诊断结束 ---")

if __name__ == '__main__':
    run_diagnostics()