#! /usr/bin/env python
# coding=utf-8

import os
import cv2
import random
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import core.utils as utils
from core.config import cfg
from core.config import args
import time
import math
#import pdb # 导入 pdb ---代码调试
from numba import jit

class Dataset(object):
    """implement Dataset here"""
    #有参构造函数  传入的数据类型决定过程 'train' --训练， ' test' -- 测试
    def __init__(self, dataset_type):
        self.annot_path  = cfg.TRAIN.ANNOT_PATH if dataset_type == 'train' else cfg.TEST.ANNOT_PATH
        self.input_sizes = cfg.TRAIN.INPUT_SIZE if dataset_type == 'train' else cfg.TEST.INPUT_SIZE
        self.batch_size  = cfg.TRAIN.BATCH_SIZE if dataset_type == 'train' else cfg.TEST.BATCH_SIZE
        self.data_aug    = cfg.TRAIN.DATA_AUG   if dataset_type == 'train' else cfg.TEST.DATA_AUG
        self.data_train_flag = True if dataset_type == 'train' else False #标记是不是训练过程 

        self.train_input_sizes = cfg.TRAIN.INPUT_SIZE
        self.strides = np.array(cfg.YOLO.STRIDES)
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.anchors = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.max_bbox_per_scale = 150

        self.annotations = self.load_annotations(dataset_type)
        self.num_samples = len(self.annotations)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size)) #step的数量 / iteration的数量
        self.batch_count = 0
        
        
        
    #加载器: 加载对应文件的每一个样本数据
    def load_annotations(self, dataset_type):
        #pdb.set_trace()
        with open(self.annot_path, 'r') as f:
            txt = f.readlines()
            annotations = [line.strip() for line in txt if line.strip()]
        np.random.shuffle(annotations) #随机打乱顺序 不按序排列 增强模型的泛化能力
        print('################### 加载有效标注总数:', len(annotations)) 
        return annotations

    def __iter__(self):
        return self

    def __next__(self):
        with tf.device('/cpu:0'):
            self.train_input_size = random.choice(self.train_input_sizes) #随机选择一个输入尺寸来进行压缩 比如416
            self.train_output_sizes = self.train_input_size // self.strides #输出的图像有三个尺度: [52,26,13] 广播机制

            batch_image = np.zeros((self.batch_size, self.train_input_size, self.train_input_size, 3))
            batch_clean_image = np.zeros((self.batch_size, self.train_input_size, self.train_input_size, 3))
            #下面三行代码是小，中，大预测框信息  输出和yolo神经网络的输出完全已知 数字5表示: 
            batch_label_sbbox = np.zeros((self.batch_size, self.train_output_sizes[0], self.train_output_sizes[0],
                                          self.anchor_per_scale, 5 + self.num_classes))
            batch_label_mbbox = np.zeros((self.batch_size, self.train_output_sizes[1], self.train_output_sizes[1],
                                          self.anchor_per_scale, 5 + self.num_classes))
            batch_label_lbbox = np.zeros((self.batch_size, self.train_output_sizes[2], self.train_output_sizes[2],
                                          self.anchor_per_scale, 5 + self.num_classes))
            
            #这下面三行代码是存放真实物体所在位置的信息, 用来临时存放从文件中读出的真实框的x,y,w,h，和yolo输出不匹配
            batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
            batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
            batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))

            num = 0
            if self.batch_count < self.num_batchs:
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num
                    if index >= self.num_samples: index -= self.num_samples
                    annotation = self.annotations[index]
                    
                    # --- 核心修改 2：在这里接收返回值并检查 ---
                    model_image_processed, bboxes_processed, clean_image_processed = self.parse_annotation(annotation)
                    
                    # 如果返回的是None，说明图片处理失败，就跳过这次循环去取下一张图片
                    if model_image_processed is None:
                        # 为了保证每个batch都能被填满，我们简单地把坏数据放到列表末尾，然后尝试下一个
                        #self.annotations.append(self.annotations.pop(index)) 
                        continue
                    # --- 检查结束 ---
                        
                    label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(bboxes_processed)

                    batch_image[num, :, :, :] = model_image_processed
                    batch_clean_image[num, :, :, :] = clean_image_processed
                    batch_label_sbbox[num, :, :, :, :] = label_sbbox
                    batch_label_mbbox[num, :, :, :, :] = label_mbbox
                    batch_label_lbbox[num, :, :, :, :] = label_lbbox
                    batch_sbboxes[num, :, :] = sbboxes
                    batch_mbboxes[num, :, :] = mbboxes
                    batch_lbboxes[num, :, :] = lbboxes
                    num += 1
                self.batch_count += 1
                return batch_image, batch_label_sbbox, batch_label_mbbox, batch_label_lbbox, \
                       batch_sbboxes, batch_mbboxes, batch_lbboxes, batch_clean_image
            else:
                self.batch_count = 0
                np.random.shuffle(self.annotations) #非常重要的一步：在开始新一天的工作前，大厨会把整个订单列表（self.annotations）的顺序完全打乱。这确保了在下一个Epoch 中，模型看到的数据顺序是随机的，这对于防止模型“背答案”、提高泛化能力至关重要。
                raise StopIteration #循环结束的标志

   
    #三个经典数据增强操作
    #random_horizontal_flip： 有 50% 的概率，将整张图片从左到右水平翻转（像照镜子一样）教会模型识别物体的对称性             
    def random_horizontal_flip(self, images, bboxes):
        if random.random() < 0.5:
            _, w, _ = images[0].shape
            images = [image[:, ::-1, :] for image in images]
            bboxes[:, [0,2]] = w - bboxes[:, [2,0]]
        return images, bboxes

    # random_crop: 模拟物体在图片中不同的位置和遮挡情况，让模型学会关注物体本身，而不是它在图片中的特定位置。
    def random_crop(self, images, bboxes):
        if random.random() < 0.5:
            h, w, _ = images[0].shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]
            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = min(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = min(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))
            images = [image[crop_ymin : crop_ymax, crop_xmin : crop_xmax] for image in images]
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin
        return images, bboxes

    #random_translate： 做什么 ✥：有 50% 的概率，将图片里的所有内容在水平和垂直方向上随机移动一小段距离。同样，它会确保移动后的所有物体依然完整地保留在画面内。移动后，它会更新所有边界框的坐标，以匹配物体的新位置。图像被移开的区域通常会变黑。
    def random_translate(self, images, bboxes):
        if random.random() < 0.5:
            h, w, _ = images[0].shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]
            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))
            M = np.array([[1, 0, tx], [0, 1, ty]])
            images = [cv2.warpAffine(image, M, (w, h)) for image in images]
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty
        return images, bboxes
    
   
    print(f"args.vocfog_traindata_dir: {args.vocfog_traindata_dir}")
    
    #重点在这里 这里采用了混合数据训练方法 把清晰的图片和有雾的图片混合在一起作为训练数据
#     def parse_annotation(self, annotation):
#         #pdb.set_trace()
#         line = annotation.split()
#         image_path = line[0].replace('\\', '/')
        
#         # --- 核心修改 1：在这里加上对坏图片的检查 ---
#         if not os.path.exists(image_path):
#             return None, None, None
        
#         clean_image = cv2.imread(image_path) #这里的image就是clean image
#         if clean_image is None:
#             return None, None, None
#         # --- 检查结束 ---

#         base_name = os.path.basename(image_path)
#         image_name_only, image_ext = os.path.splitext(base_name)

#         bboxes = np.array([list(map(lambda x: int(float(x)), box.split(','))) for box in line[1:]])
        
#         model_input_image = np.copy(clean_image)
#         #选取人工增雾并进行数据增强的概率: 2/3(输出的既有有雾的图片 也有清晰的图片) 选取没有人工增雾并进行数据增强的概率: 1/3 (有雾的图片就是清晰的图片)
#         if random.randint(0, 2) > 0 and self.data_train_flag:
#             beta = random.randint(0, 9)
#             beta = 0.01 * beta + 0.05
            
#             foggy_filename = f"{image_name_only}_{beta:.2f}{image_ext}" #找foggy image
#             img_name = os.path.join(args.vocfog_traindata_dir, foggy_filename)
#             print(f"尝试加载雾化图片: {img_name}") # 这是最关键的打印信息！

#             foggy_image_loaded = cv2.imread(img_name)
            
#             if foggy_image_loaded is None:
#                 print(f"WARN: 雾化图片无法读取或不存在: {img_name}")
#                 return None, None, None
            
#             model_input_image = foggy_image_loaded
            
#             #对图片和bbox进行同步数据增强
#         if self.data_aug and len(bboxes) > 0:
#             image_list = [clean_image,model_input_image]
#             # 依次调用同步的增强函数
#             image_list, bboxes = self.random_horizontal_flip(image_list, bboxes)
#             image_list, bboxes = self.random_crop(image_list, bboxes)
#             image_list, bboxes = self.random_translate(image_list, bboxes)
                 
#             # 从列表中解包出处理好的图片
#             clean_image, model_input_image = image_list
        
#         model_input_processed, bboxes_processed = utils.image_preporcess(np.copy(model_input_image), [self.train_input_size, self.train_input_size], np.copy(bboxes))
#         clean_image_processed, _ = utils.image_preporcess(np.copy(clean_image), [self.train_input_size, self.train_input_size], np.copy(bboxes_processed))
#         return model_input_processed, bboxes_processed, clean_image_processed
    
#     #微调阶段定制 
    def parse_annotation(self, annotation):
        """
        这个版本专门用于微调阶段：
        1. 直接加载输入的真实雾图（来自RTTS）。
        2. 不再需要加载清晰图和合成雾图的复杂逻辑。
        3. 依然支持同步的数据增强。
        """
        try:
            line = annotation.split()
            # 现在的 image_path 直接就是我们要训练的真实雾图
            image_path = line[0].replace('\\', '/')
            
            if not os.path.exists(image_path):
                print(f"WARN: 找不到图片文件: {image_path}")
                return None, None, None
            
            # 1. 直接加载 RTTS 的雾图作为模型输入
            model_input_image = cv2.imread(image_path)
            if model_input_image is None:
                print(f"WARN: 无法读取图片文件: {image_path}")
                return None, None, None

            # 2. 我们不再有清晰图了，但为了保持数据格式一致，
            #    可以简单地用输入图自己作为 "clean_image" 占位符。
            #    在微调时，我们会禁用恢复损失，所以这个占位符不会被使用。
            clean_image = np.copy(model_input_image)
            
            bboxes = np.array([list(map(lambda x: int(float(x)), box.split(','))) for box in line[1:]])
            
            # 3. (核心修正) 数据增强现在直接作用于真实雾图和清晰图占位符
            if self.data_aug and len(bboxes) > 0:
                # 将占位符和真实雾图打包，进行同步增强
                image_list = [clean_image, model_input_image]
                
                image_list, bboxes = self.random_horizontal_flip(image_list, bboxes)
                image_list, bboxes = self.random_crop(image_list, bboxes)
                image_list, bboxes = self.random_translate(image_list, bboxes)
                
                clean_image, model_input_image = image_list

            # 4. 最终预处理
            model_input_processed, bboxes_processed = utils.image_preporcess(np.copy(model_input_image), [self.train_input_size, self.train_input_size], np.copy(bboxes))
            clean_image_processed, _ = utils.image_preporcess(np.copy(clean_image),[self.train_input_size, self.train_input_size], np.copy(bboxes_processed))

            return model_input_processed, bboxes_processed, clean_image_processed
            
        except Exception as e:
            print(f"处理行 '{annotation}' 时发生严重错误: {e}")
            return None, None, None
    
    #计算bbox_iou的代码(交并比)
    def bbox_iou(self, boxes1, boxes2):
        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        # Correct area calculation for xywh format
        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                 boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                 boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        return inter_area / (union_area + 1e-6)

    #将简单的真实框列表（bboxes），转换成神经网络能够直接理解和计算损失的、极其复杂的**“标准答案”矩阵**（label_sbbox等）。
    #所以，preprocess_true_boxes函数里的主循环 for bbox in bboxes:，是在一位一位地处理新生。

   # 当处理一位新生（一个bbox）时，他的“入住流程”是这样的：

   # 这位新生来到报到处。

   # 他会依次去看S、M、L三栋宿舍楼（3个尺度）。

   # 在每一栋楼里，他会找到自己对应的那个房间（由他的坐标决定），然后看看房间里的3张床（3个锚点框），并评估自己和这3张床的“匹配度”（IoU）。

   # 根据匹配度，决定这位新生应该“负责”哪几张床。

   # 因此，是**一个新生（bbox）在和多个床位（锚点框）**进行匹配，而不是‘3个新生代表3个锚点框’。
    def preprocess_true_boxes(self, bboxes):
        label = [np.zeros((self.train_output_sizes[i], self.train_output_sizes[i], self.anchor_per_scale,
                           5 + self.num_classes)) for i in range(3)]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]
        bbox_count = np.zeros((3,))
        
        for bbox in bboxes:
            # ==================== 终极安全卫士 Part 1: 检查原始数据 ====================
            # 检查bbox本身是否包含 NaN(非数字) 或 inf(无穷大)
            if np.any(np.isnan(bbox)) or np.any(np.isinf(bbox)):
                print(f"\n[!!!] 警告：发现包含 NaN/inf 的非法标注，已跳过。原始 bbox: {bbox}")
                continue
            
            bbox_coor = bbox[:4]
            bbox_class_ind = int(bbox[4]) # 确保类别ID是整数

            # 检查类别ID是否越界 (例如5个类时, ID必须在0-4之间)
            if not 0 <= bbox_class_ind < self.num_classes:
                print(f"\n[!!!] 警告：发现越界的类别ID {bbox_class_ind}，已跳过。")
                continue
            # =====================================================================

            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            # 把坐标从 [xmin, ymin, xmax, ymax] 格式，转换成 [中心x, 中心y, 宽, 高] 格式。
            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            
            # ==================== 终极安全卫士 Part 2: 检查计算结果 ====================
            # 检查转换后的宽高是否合法
            width, height = bbox_xywh[2], bbox_xywh[3]
            if width <= 0 or height <= 0:
                print(f"\n[!!!] 警告：发现非法边界框（宽/高<=0），已跳过。原始 bbox: {bbox}")
                continue
            
            # (双重保险) 再次检查，防止转换过程中出现inf/nan
            if np.any(np.isnan(bbox_xywh)) or np.any(np.isinf(bbox_xywh)):
                print(f"\n[!!!] 警告：坐标转换后出现 NaN/inf，已跳过。原始 bbox: {bbox}")
                continue
            # =====================================================================

            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]
            
            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = self.anchors[i]
                iou_scale = self.bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)
                    
                    # 增加索引裁剪保护，防止越界
                    grid_size = self.train_output_sizes[i]
                    xind, yind = np.clip([xind, yind], 0, grid_size - 1)
                    xind, yind = int(xind), int(yind)
                    
                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot
                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1
                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                # 这里也需要同样的索引裁剪保护
                grid_size = self.train_output_sizes[best_detect]
                xind, yind = np.clip([xind, yind], 0, grid_size - 1)
                xind, yind = int(xind), int(yind)
                
                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot
                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1
                
        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes
    
    def __len__(self):
        return self.num_batchs



