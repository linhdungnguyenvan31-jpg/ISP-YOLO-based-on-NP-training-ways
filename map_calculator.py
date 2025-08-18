import os
import numpy as np
import matplotlib.pyplot as plt

# ----------------- 辅助函数：判断物体尺寸类别 -----------------
def get_object_size_category(box_coords):
    """
    根据边界框的坐标，判断其属于小、中、大哪个类别。
    box_coords: 一个列表或数组，格式为 [xmin, ymin, xmax, ymax]
    """
    xmin, ymin, xmax, ymax = map(float, box_coords)
    width = xmax - xmin
    height = ymax - ymin
    area = width * height

    if area < 1024:  # 32*32
        return "small"
    elif 1024 <= area < 9216:  # 96*96
        return "medium"
    else:
        return "large"

# ----------------- 辅助函数：计算IoU -----------------
def calculate_iou(box1, box2):
    """计算两个框的IoU (xmin, ymin, xmax, ymax)"""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0.0
    return iou

# ----------------- 辅助函数：绘制P-R曲线 -----------------
def plot_pr_curve(class_name, recalls, precisions, ap, save_dir):
    """绘制并保存P-R曲线图"""
    plt.figure()
    plt.plot(recalls, precisions, marker='.', label=f'Precision-Recall Curve (AP = {ap:.4f})')
    interp_precisions = np.copy(precisions)
    for i in range(len(interp_precisions) - 2, -1, -1):
        interp_precisions[i] = max(interp_precisions[i], interp_precisions[i+1])
    plt.plot(recalls, interp_precisions, linestyle='--', label=f'Interpolated P-R Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'P-R Curve for: {class_name}')
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(save_dir, f'PR_Curve_{class_name}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"  -> P-R curve for '{class_name}' saved to {save_path}")

# ==================================================================
#               新增：mAP 分类条形图绘制函数
# ==================================================================
def plot_map_barchart(class_aps, mAP, save_dir):
    """
    绘制并保存 mAP 条形图。
    :param class_aps (dict): 包含每个类别 AP 值的字典, e.g., {'cat': 0.9, 'dog': 0.8}
    :param mAP (float): 总的 mAP 值 (0到1之间)
    :param save_dir (str): 图表保存的目录
    """
    print(f"\n--- 正在生成 mAP 分类条形图 ---")
    
    # 1. 数据准备与排序
    sorted_aps = sorted(class_aps.items(), key=lambda item: item[1])
    class_names = [item[0] for item in sorted_aps]
    ap_values = [item[1] for item in sorted_aps]

    # 2. 创建图表
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(class_names, ap_values, color='royalblue')

    # 3. 美化图表
    ax.set_xlabel('Average Precision', fontsize=12)
    ax.set_title(f'mAP = {mAP*100:.2f}%', fontsize=16)
    ax.set_xlim(0, 1.05)
    ax.tick_params(axis='y', length=0)

    # 4. 在条形图上添加数值标签
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2, f'{width:.2f}', va='center', ha='left')

    fig.tight_layout()

    # 5. 保存并显示图表
    save_path = os.path.join(save_dir, 'mAP_barchart.png')
    plt.savefig(save_path, dpi=300)
    print(f"  -> mAP条形图已成功保存到: {save_path}")
    plt.close() # 关闭图形，防止后续的plt.show()再次显示
# ==================================================================

# ----------------- 辅助函数：核心AP计算逻辑 -----------------
def _calculate_ap(ground_truths_filtered, predictions_all, iou_threshold):
    """
    一个通用的AP计算函数
    """
    # (您的 _calculate_ap 函数代码保持不变)
    for gt in ground_truths_filtered:
        gt['matched'] = False
    gt_classes = set([gt['class'] for gt in ground_truths_filtered])
    predictions_filtered = [pred for pred in predictions_all if pred['class'] in gt_classes]
    predictions_filtered.sort(key=lambda x: x['confidence'], reverse=True)
    
    tp = 0
    fp = 0
    recalls = []
    precisions = []
    
    total_gt_positives = len(ground_truths_filtered)
    if total_gt_positives == 0:
        return 0, [], []

    for pred in predictions_filtered:
        image_id = pred['image_id']
        gt_boxes_on_img = [box for box in ground_truths_filtered if box.get('image_id') == image_id and box['class'] == pred['class']]
        best_iou, best_gt_idx = 0, -1
        for i, gt_box in enumerate(gt_boxes_on_img):
            iou = calculate_iou(pred['coords'], gt_box['coords'])
            if iou > best_iou:
                best_iou, best_gt_idx = iou, i
        
        if best_iou >= iou_threshold and not gt_boxes_on_img[best_gt_idx]['matched']:
            tp += 1
            gt_boxes_on_img[best_gt_idx]['matched'] = True
        else:
            fp += 1
        
        recall = tp / total_gt_positives
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recalls.append(recall)
        precisions.append(precision)
        
    recalls_interp = np.concatenate(([0.], recalls, [1.]))
    precisions_interp = np.concatenate(([0.], precisions, [0.]))
    for i in range(len(precisions_interp) - 2, -1, -1):
        precisions_interp[i] = max(precisions_interp[i], precisions_interp[i+1])
    ap = 0
    for i in range(len(recalls_interp) - 1):
        if recalls_interp[i+1] != recalls_interp[i]:
            ap += (recalls_interp[i+1] - recalls_interp[i]) * precisions_interp[i+1]
            
    return ap, recalls, precisions

# ----------------- 主函数：计算mAP及其他指标 -----------------
def calculate_map(ground_truth_dir, predicted_dir, iou_threshold=0.5, draw_plot=True):
    """
    根据生成的真值和预测文件夹，计算mAP，并按类别和尺寸分析AP
    """
    print(f"--- 开始计算mAP, IoU阈值为: {iou_threshold} ---")
    
    # 准备保存绘图的文件夹
    plot_save_dir = os.path.join(os.path.dirname(ground_truth_dir), 'plots')
    if draw_plot and not os.path.exists(plot_save_dir):
        os.makedirs(plot_save_dir)

    # 1. 解析所有文件
    ground_truths_all = []
    predictions_all = []
    all_classes = set()
    
    # (您的文件解析代码保持不变)
    for filename in os.listdir(ground_truth_dir):
        image_id = filename.split('.')[0]
        with open(os.path.join(ground_truth_dir, filename), 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_name, coords = parts[0], list(map(float, parts[1:]))
                size_tag = get_object_size_category(coords)
                ground_truths_all.append({'image_id': image_id, 'class': class_name, 'coords': coords, 'size': size_tag, 'matched': False})
                all_classes.add(class_name)

    for filename in os.listdir(predicted_dir):
        image_id = filename.split('.')[0]
        with open(os.path.join(predicted_dir, filename), 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_name, confidence, coords = parts[0], float(parts[1]), list(map(float, parts[2:]))
                predictions_all.append({'image_id': image_id, 'class': class_name, 'confidence': confidence, 'coords': coords})
                all_classes.add(class_name)
    
    # 2. 按类别计算AP
    print("\n--- 按类别计算 AP (Average Precision by Class) ---")
    aps_by_class = {}
    all_classes = sorted(list(all_classes))

    for class_name in all_classes:
        class_gts = [gt for gt in ground_truths_all if gt['class'] == class_name]
        ap, recalls, precisions = _calculate_ap(class_gts, predictions_all, iou_threshold)
        aps_by_class[class_name] = ap
        print(f"  -> AP for class '{class_name}': {ap:.4f}")

        if draw_plot and recalls:
            plot_pr_curve(class_name, recalls, precisions, ap, plot_save_dir)

    # 3. 计算mAP
    mean_ap = np.mean(list(aps_by_class.values())) if aps_by_class else 0
    print(f"\n--- mAP (mean Average Precision): {mean_ap:.4f} ---")
    
    # ===================================================
    #   新增：在这里调用条形图绘制函数 (最合适的位置！)
    # ===================================================
    if draw_plot and aps_by_class:
        plot_map_barchart(aps_by_class, mean_ap, plot_save_dir)
    
    # 4. 按尺度计算AP
    print("\n--- 按物体尺寸计算 AP (Average Precision by Object Size) ---")
    aps_by_size = {}
    size_groups = ["small", "medium", "large"]

    for size_name in size_groups:
        size_gts = [gt for gt in ground_truths_all if gt['size'] == size_name]
        ap, _, _ = _calculate_ap(size_gts, predictions_all, iou_threshold)
        aps_by_size[size_name] = ap
        print(f"  -> AP for '{size_name}' objects: {ap:.4f}")
    
    return mean_ap, aps_by_class, aps_by_size