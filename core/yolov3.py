#! /usr/bin/env python
# coding=utf-8


import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import core.utils as utils
import core.common as common
import core.backbone as backbone
from core.config import cfg
import time


class YOLOV3(object):
    """Implement tensoflow yolov3 here"""
    def __init__(self, input_data, trainable, input_data_clean, defog_A=None, IcA=None):

        self.trainable        = trainable
        self.classes          = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_class        = len(self.classes)
        self.strides          = np.array(cfg.YOLO.STRIDES)
        self.anchors          = utils.get_anchors(cfg.YOLO.ANCHORS)
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.iou_loss_thresh  = cfg.YOLO.IOU_LOSS_THRESH
        self.upsample_method  = cfg.YOLO.UPSAMPLE_METHOD
        self.isp_flag = cfg.YOLO.ISP_FLAG


        try:
            self.conv_lbbox, self.conv_mbbox, self.conv_sbbox,self.recovery_loss= \
                self.__build_nework(input_data, self.isp_flag, input_data_clean, defog_A, IcA)
        except Exception as e:
            print(f"Error occurred while building the YOLOv3 network: {e}")
            raise NotImplementedError("Can not build up yolov3 network!")
 
        #提取预测框的特征 每个尺度对应不同的anchor 每个anchor下面有三个不同形状的锚框  第一个锚框--小物体  第三个锚框 -- 大物体
        with tf.variable_scope('pred_sbbox'):
            self.pred_sbbox = self.decode(self.conv_sbbox, self.anchors[0], self.strides[0])

        with tf.variable_scope('pred_mbbox'):
            self.pred_mbbox = self.decode(self.conv_mbbox, self.anchors[1], self.strides[1])

        with tf.variable_scope('pred_lbbox'):
            self.pred_lbbox = self.decode(self.conv_lbbox, self.anchors[2], self.strides[2])

    
    #这个“从上到下，层层融合”的结构就是YOLOv3中FPN（特征金字塔网络）的经典实现，它使得模型能够同时看到图片的高层“含义”和底层“细节”
                            #(有雾图片)            #无雾 清晰的图片
    def __build_nework(self, input_data, isp_flag, input_data_clean, defog_A, IcA): #专家自动调参网络+yolov3神经网络

        filtered_image_batch = input_data
        self.filter_params = input_data
        filter_imgs_series = []
        if isp_flag:#isp_flag（Image Signal Processing Flag）决定了是否启用这个“AI Photoshop”---就是原作者使用的迷你cnn自动调整超参数 功能。如果关闭，就走常规流程。
            # start_time = time.time()
            
            #第二部: 定义专家自动调超参数的方式
            with tf.variable_scope('extract_parameters_2'):
                input_data = tf.image.resize_images(input_data, [256, 256], method=tf.image.ResizeMethod.BILINEAR) #首先，代码将输入图片缩小到 256x256（为了快速处理），然后喂给一个名为 common.extract_parameters_2 的小型辅助神经网络。
                filter_features = common.extract_parameters_2(input_data, cfg, self.trainable) # filter_features。你可以把 filter_features 理解为是AI专家对这张图片“诊断”后，给出的一份“P图参数建议报告”。这份报告里包含了一系列数值，暗示了这张图应该如何调整。

            # filter_features = tf.random_normal([1, 15], 0.5, 0.1)
            
            #第三步: 取出不同的工具箱 让不同的工具箱接受不同的超参数 来进行工作
            filters = cfg.filters
            filters = [x(filtered_image_batch, cfg) for x in filters]
            filter_parameters = []
            
            #第4步：流水线作业，依次P图
            for j, filter in enumerate(filters):
                with tf.variable_scope('filter_%d' % j):
                    print('    creating filter:', j, 'name:', str(filter.__class__), 'abbr.',
                          filter.get_short_name())
                    print('      filter_features:', filter_features.shape)

                    filtered_image_batch, filter_parameter = filter.apply(
                        filtered_image_batch, filter_features, defog_A, IcA) #filter.apply 函数会输出处理后焕然一新的图片 filtered_image_batch，这张新图片会立即成为流水线下一个工具的输入。
                    
                    #把第一张图片存到filter_imgs_series里面，并把这张图片作为下一次的输入 同时，它还会输出这次操作所用的具体数值 filter_parameter（比如1.2），并存起来。
                    filter_parameters.append(filter_parameter)
                    filter_imgs_series.append(filtered_image_batch)
                    print('      output:', filtered_image_batch.shape)

            self.filter_params = filter_parameters  # 当图片走完整个“P图”流水线后，所有步骤中用到的具体参数（比如亮度+1.2, 对比度+0.9等）会被统一收集起来，
            # end_time = time.time()
            # print('filters所用时间：', end_time - start_time)
        # input_data_shape = tf.shape(input_data)
        # batch_size = input_data_shape[0]
        
        
        
        
        
        # recovery_loss 是在给之前的AI-photoshop打分 
        recovery_loss = tf.reduce_sum(tf.pow(filtered_image_batch - input_data_clean, 2.0))#/(2.0 * batch_size)
        self.image_isped = filtered_image_batch
        self.filter_imgs_series = filter_imgs_series
        input_data = filtered_image_batch #精修图
        
        #主干道加工 route1---52*52小物体检测 route2 --- 26*26中等物体检测 ----13*13  大物体检测 升采样: 把原来的图像分辨率有13*13 --- 26*26 重建图像分辨率
        route_1, route_2, input_data = backbone.darknet53(input_data, self.trainable)

        #定做大物体检测的检测头
        #中间这四个数字表示的含义 -- 联系卷积层
        #好的，这行代码是定义一个卷积层（Convolutional Layer），括号里的四个参数 (1, 1, 1024, 512) 是定义这个卷积层的**卷积核（Kernel）或过滤器（Filter）**的形状和数量。

        # 这四个数字的含义依次是：

        # 卷积核高度 (Kernel Height)

        # 卷积核宽度 (Kernel Width)

        # 输入通道数 (Input Channels)

        # 输出通道数 (Output Channels / Filter Count)
        #以第一个为例
        #         我们以您给出的第一行代码为例：
        # common.convolutional(input_data, (1, 1, 1024, 512), ...)

        # 1, 1 (卷积核高和宽)：这定义了卷积核的大小是 1x1。这是一个特殊的卷积，通常用来在不改变特征图尺寸的情况下，对通道数进行降维或升维，可以理解为对不同通道的信息进行融合和重组。

        # 1024 (输入通道数)：这必须与输入数据 input_data 的通道数完全匹配。它告诉卷积层：“我准备处理的数据，是有1024个通道的特征图。”

        # 512 (输出通道数)：这定义了该卷积层要使用 512个 不同的卷积核。每个卷积核都会在输入数据上进行运算，并产生一个输出特征图。因此，经过这个卷积层处理后，输出的特征图的通道数就会从1024变为512。

        # 总结一下：这行代码的作用就是用512个1x1的卷积核，将一个通道数为1024的输入特征图，转换成一个通道数为512的输出特征图。
        input_data = common.convolutional(input_data, (1, 1, 1024,  512), self.trainable, 'conv52')
        input_data = common.convolutional(input_data, (3, 3,  512, 1024), self.trainable, 'conv53')
        input_data = common.convolutional(input_data, (1, 1, 1024,  512), self.trainable, 'conv54')
        input_data = common.convolutional(input_data, (3, 3,  512, 1024), self.trainable, 'conv55')
        input_data = common.convolutional(input_data, (1, 1, 1024,  512), self.trainable, 'conv56')

        conv_lobj_branch = common.convolutional(input_data, (3, 3, 512, 1024), self.trainable, name='conv_lobj_branch')
        conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 1024, 3*(self.num_class + 5)),
                                          trainable=self.trainable, name='conv_lbbox', activate=False, bn=False)

        #13*13的特征图进行上采样，得到了26*26 的图片 重建分辨率 ---和已经有的route2信息结合起来 构建中等物体检测头
        input_data = common.convolutional(input_data, (1, 1,  512,  256), self.trainable, 'conv57')
        input_data = common.upsample(input_data, name='upsample0', method=self.upsample_method)

        with tf.variable_scope('route_1'):
            input_data = tf.concat([input_data, route_2], axis=-1)

        input_data = common.convolutional(input_data, (1, 1, 768, 256), self.trainable, 'conv58')
        input_data = common.convolutional(input_data, (3, 3, 256, 512), self.trainable, 'conv59')
        input_data = common.convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv60')
        input_data = common.convolutional(input_data, (3, 3, 256, 512), self.trainable, 'conv61')
        input_data = common.convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv62')

        conv_mobj_branch = common.convolutional(input_data, (3, 3, 256, 512),  self.trainable, name='conv_mobj_branch' )
        conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 512, 3*(self.num_class + 5)),
                                          trainable=self.trainable, name='conv_mbbox', activate=False, bn=False)

        #26*26的特征图进行上采样，得到了52*52的图片 重建分辨率 ---和已经有的route1信息结合起来 构建中等物体检测头
        input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv63')
        input_data = common.upsample(input_data, name='upsample1', method=self.upsample_method)

        with tf.variable_scope('route_2'):
            input_data = tf.concat([input_data, route_1], axis=-1)

        input_data = common.convolutional(input_data, (1, 1, 384, 128), self.trainable, 'conv64')
        input_data = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, 'conv65')
        input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv66')
        input_data = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, 'conv67')
        input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv68')

        conv_sobj_branch = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, name='conv_sobj_branch')
        conv_sbbox = common.convolutional(conv_sobj_branch, (1, 1, 256, 3*(self.num_class + 5)),
                                          trainable=self.trainable, name='conv_sbbox', activate=False, bn=False)

        
        return conv_lbbox, conv_mbbox, conv_sbbox,recovery_loss

    #decode的任务是把我们看不懂的conv_outputs转成我们能看懂的 具体的边界框信息
    def decode(self, conv_output, anchors, stride):
        """
        return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
               contains (x, y, w, h, score, probability)
        """
        #conv_output: 具体来说是YOLO的预测头/Head
        conv_shape       = tf.shape(conv_output)
        batch_size       = conv_shape[0]
        output_size      = conv_shape[1]
        anchor_per_scale = len(anchors)

        #第一步:重组输出格式 按照输出的格式来重组
        conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, anchor_per_scale, 5 + self.num_class))

        conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
        conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
        conv_raw_conf = conv_output[:, :, :, :, 4:5]
        conv_raw_prob = conv_output[:, :, :, :, 5: ]

        #第二步制作坐标参考网格 制作出一系列的参考网格 :因为神经网络预测的中心点dxdy是一个偏移量 需要有参考基准 这个基准就是它所在的网格单元的左上角坐标
        y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
        x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])

        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchor_per_scale, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)
        
        #这一步;破译边界框
        #这一步主要是在破译边界框 --- 其中x和y是通过sigmoid函数放缩到0-1之间 确保在网格里面
        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * stride# 最后sigmoid函数步长 --- 回到原来的图像中
        pred_wh = (tf.exp(conv_raw_dwdh) * anchors) * stride #指数函数: 由于wh -- 神经网络返回的是比例:宽高比 此时要成上锚框相对于网格的比例(乘回去） 回到正常的比例 然后再乘以锚框
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1) #把xywh拼起来

        pred_conf = tf.sigmoid(conv_raw_conf) #置信度破译:直接用 sigmoid 函数将置信度和每个类别的原始分数，都转换成 0~1 之间的概率值。
        pred_prob = tf.sigmoid(conv_raw_prob)

        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1) # 最后，将所有解码完成的、我们能看懂的信息——边界框、置信度、类别概率——全部拼接起来，作为最终的输出。

    # 这是一种分类损失函数。在目标检测中，它主要用来判断一个区域“是不是物体”（置信度损失）以及“是哪一类物体”（分类损失）。
    #它是如何工作的？ Focal Loss像一个聪明的老师批改试卷：

# 对于模型很容易就判断对的样本（比如一眼就看出来的背景），它会给一个非常小的权重。相当于老师说：“这题太简单了，答对了也不加多少分。”

# 对于模型判断错或者不确定的样本（比如把背景错当成物体，或者没找到物体），它会给一个非常大的权重。相当于老师说：“这题你都错了，问题很大！要重罚，让你印象深刻！” 是否有物体
#让正样本参与计算
    def focal(self, target, actual, alpha=1, gamma=2): #gamma 参数：就是“惩罚力度”，gamma越大，对难样本的关注就越多。
        focal_loss = alpha * tf.pow(tf.abs(target - actual), gamma)
        return focal_loss

    #bbox_giou (Generalized IoU) —— “不仅看贴合，还看方向”的升级版损失函数带来的好处：即使两个框完全不重叠，模型也能通过最小化这个“空白区域”的惩罚，得到一个清晰的梯度信号，知道应该朝哪个方向移动预测框才能离真实框更近。
    def bbox_giou(self, boxes1, boxes2):

        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                            tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
        boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                            tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = inter_area / union_area

        enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
        enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
        enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
        enclose_area = enclose[..., 0] * enclose[..., 1]
        giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

        return giou

    #IoU是用来计算两个边界框重合度的通用工具，是目标检测领域的“硬通货”。
    def bbox_iou(self, boxes1, boxes2):

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = 1.0 * inter_area / union_area

        return iou

    def loss_layer(self, conv, pred, label, bboxes, anchors, stride):

        conv_shape  = tf.shape(conv)
        batch_size  = conv_shape[0]
        output_size = conv_shape[1]
        input_size  = stride * output_size
        conv = tf.reshape(conv, (batch_size, output_size, output_size,
                                 self.anchor_per_scale, 5 + self.num_class))
        conv_raw_conf = conv[:, :, :, :, 4:5]
        conv_raw_prob = conv[:, :, :, :, 5:]

        pred_xywh     = pred[:, :, :, :, 0:4]
        pred_conf     = pred[:, :, :, :, 4:5]

        label_xywh    = label[:, :, :, :, 0:4]
        respond_bbox  = label[:, :, :, :, 4:5]
        label_prob    = label[:, :, :, :, 5:]

        #giou = self.bbox_giou(...): 老师拿出我们之前讨论过的 GIoU 这把高级尺子，来计算学生画的框和标准答案框的“贴合分数”。老师的思路: “我先看看这个学生画的框 (pred_xywh)，和我的标准答案框 (label_xywh) 贴合得怎么样。”
        
        giou = tf.expand_dims(self.bbox_giou(pred_xywh, label_xywh), axis=-1)
        input_size = tf.cast(input_size, tf.float32)
        
        #计算giou_loss的损失函数
        bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
        giou_loss = respond_bbox * bbox_loss_scale * (1- giou)

        iou = self.bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
        max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

        respond_bgd = (1.0 - respond_bbox) * tf.cast( max_iou < self.iou_loss_thresh, tf.float32 )

        conf_focal = self.focal(respond_bbox, pred_conf)

        

        ## 题型二：判断题 - 这里到底有没有物体？ (conf_loss)
        #这是置信度损失 (Confidence/Objectness Loss)，评价的是模型判断“有物体”还是“没物体”的能力。这是最复杂的一道题。
        conf_loss = conf_focal * (
                respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
                +
                respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
        )

        #老师的思路: “这道题的前提是，必须先答对上一道判断题。只有在标准答案里确实有物体的地方 (respond_bbox 为 1)，我才批改这道选择题。”
        prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)
        
        #汇总总分
        giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
        prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4]))

        return giou_loss, conf_loss, prob_loss



    def compute_loss(self, label_sbbox, label_mbbox, label_lbbox, true_sbbox, true_mbbox, true_lbbox):

        with tf.name_scope('smaller_box_loss'):
            loss_sbbox = self.loss_layer(self.conv_sbbox, self.pred_sbbox, label_sbbox, true_sbbox,
                                         anchors = self.anchors[0], stride = self.strides[0])

        with tf.name_scope('medium_box_loss'):
            loss_mbbox = self.loss_layer(self.conv_mbbox, self.pred_mbbox, label_mbbox, true_mbbox,
                                         anchors = self.anchors[1], stride = self.strides[1])

        with tf.name_scope('bigger_box_loss'):
            loss_lbbox = self.loss_layer(self.conv_lbbox, self.pred_lbbox, label_lbbox, true_lbbox,
                                         anchors = self.anchors[2], stride = self.strides[2])

        with tf.name_scope('giou_loss'):
            giou_loss = loss_sbbox[0] + loss_mbbox[0] + loss_lbbox[0]

        with tf.name_scope('conf_loss'):
            conf_loss = loss_sbbox[1] + loss_mbbox[1] + loss_lbbox[1]

        with tf.name_scope('prob_loss'):
            prob_loss = loss_sbbox[2] + loss_mbbox[2] + loss_lbbox[2]

        with tf.name_scope('recovery_loss'):
            recovery_loss = self.recovery_loss

        return giou_loss, conf_loss, prob_loss,recovery_loss


