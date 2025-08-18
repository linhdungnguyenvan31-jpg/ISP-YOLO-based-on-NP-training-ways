#! /usr/bin/env python
# coding=utf-8
# 每次通过命令行传入 传入这个文件所在的具体位置
# 格式：python [你要运行的脚本] --参数名 "你的新路径"

from easydict import EasyDict as edict
from filters import *
import argparse
#print("--- 评估脚本开始运行！ ---")
parser = argparse.ArgumentParser(description='')#创建命令行参数的对象
parser.add_argument('--exp_num', dest='exp_num', type=str, default='101', help='current experiment number')
parser.add_argument('--epoch_first_stage', dest='epoch_first_stage', type=int, default=0, help='# of epochs')#第一个阶段: 热身 --- 冻结主干，只训练头部，快速适应
parser.add_argument('--epoch_second_stage', dest='epoch_second_stage', type=int, default=20, help='# of epochs')#第二个阶段: 正式训练 ---（微调）：解冻主干，用小学习率整体训练，精细打磨模型性能 
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU') # 相当于把use_gpu这个变量幅值成1
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--exp_dir', dest='exp_dir', default='./experiments', help='models are saved here')
parser.add_argument('--gpu_id', dest='gpu_id', type=str, default='0', help='if use gpu, use gpu device id')
parser.add_argument('--ISP_FLAG', dest='ISP_FLAG', type=bool, default=True, help='whether use DIP Module')  #数字信号模块 用于处理图像 这里会结合自动调整超参数的
parser.add_argument('--fog_FLAG', dest='fog_FLAG', type=bool, default=True, help='whether use Hybrid data training') #原作者在训练时用了混合数据方法 提高模型的泛化能力
parser.add_argument('--vocfog_traindata_dir', dest='vocfog_traindata_dir', default='./data/rttsimages/train/JPEGImages', #下面三行都是人工增雾处理的
                    help='the dir contains ten levels synthetic foggy images') #数据集 有人工增雾的图片(jpg格式)
parser.add_argument('--vocfog_valdata_dir', dest='vocfog_valdata_dir', default='./data/rttsimages/val/JPEGImages/',
                    help='the dir contains ten levels synthetic foggy images')
parser.add_argument('--vocfog_testdata_dir', dest='vocfog_testdata_dir', default='./data/rttsimages/test/JPEGImages/',
                    help='the dir contains ten levels synthetic foggy images')
parser.add_argument('--train_path', dest='train_path', default='./data/rtts/rtts_train.txt', help='folder of the training data') #.txt格式
parser.add_argument('--val_path', dest='val_path', default='./data/rtts/rtts_val.txt', help='folder of the validate data')
parser.add_argument('--test_path', dest='test_path', default='./data/rtts/rtts_test.txt', help='folder of the training data')
parser.add_argument('--class_name', dest='class_name', default='./data/classes/vocfog.names', help='folder of the training data') #作者自定义的类别 5个
parser.add_argument('--WRITE_IMAGE_PATH', dest='WRITE_IMAGE_PATH', default='./experiments/exp_101/checkpoint/detection_results/', help='folder of the training data') 
parser.add_argument('--WEIGHT_FILE', dest='WEIGHT_FILE', default='./experiments/exp_101/checkpoint/yolov3_test_loss=8.6692.ckpt-18', help='folder of the training data') #你训练好模型的文件路径
parser.add_argument('--pre_train', dest='pre_train', default='./experiments/exp_101/checkpoint/yolov3_test_loss=0.0659.ckpt-26', help='the path of pretrained models if is not null. not used for now')
# we trained our model from scratch.





args = parser.parse_args() #激活命令行参数



__C                             = edict()
# Consumers can get config by: from config import cfg

cfg                             = __C

###########################################################################
# Filter Parameters  
###########################################################################
#输入一张图 -> AI模型预测出15个参数 -> 系统根据配置，将这15个参数分配给6个不同的滤镜 -> 滤镜根据分配到的参数值和预设的范围处理图像 -> 输出最终效果图。

#cfg.filters, 作用：定义了图像处理的步骤和顺序。一张图片输入后，会依次经过以下滤镜（Filter）处理
cfg.filters = [
    DefogFilter, ImprovedWhiteBalanceFilter,  GammaFilter,
    ToneFilter, ContrastFilter, UsmFilter
]
cfg.num_filter_parameters = 15 #总参数15个 第0个参数: defog 第1-3个参数: wb 第四个:gamma 第五个-第十二个: tone(主导) 第十三个: contrast 第十四个: use

cfg.defog_begin_param = 0

cfg.wb_begin_param = 1
cfg.gamma_begin_param = 4
cfg.tone_begin_param = 5
cfg.contrast_begin_param = 13
cfg.usm_begin_param = 14

#作用：定义了每个参数的有效取值范围。AI模型通常输出的是归一化（比如0到1之间)
cfg.curve_steps = 8
cfg.gamma_range = 3
cfg.exposure_range = 3.5
cfg.wb_range = 1.1
cfg.color_curve_range = (0.90, 1.10)
cfg.lab_curve_range = (0.90, 1.10)
cfg.tone_curve_range = (0.5, 2)
cfg.defog_range = (0.1, 1.0)
cfg.usm_range = (0.0, 5)



# Masking is DISABLED
cfg.masking = False
cfg.minimum_strength = 0.3
cfg.maximum_sharpness = 1
cfg.clamp = False

###########################################################################
# CNN Parameters
###########################################################################
cfg.source_img_size = 64
cfg.base_channels = 32
cfg.dropout_keep_prob = 0.5
# G and C use the same feed dict?
cfg.share_feed_dict = True
cfg.shared_feature_extractor = True
cfg.fc1_size = 128
cfg.bnw = False
# number of filters for the first convolutional layers for all networks
#                      (stochastic/deterministic policy, critic, value)
cfg.feature_extractor_dims = 4096

###########################################################################

# YOLO options
__C.YOLO                        = edict()

# Set the class name
__C.YOLO.CLASSES                = args.class_name
__C.YOLO.ANCHORS                = "./data/anchors/coco_anchors.txt" #聚类的九个预测框的坐标
__C.YOLO.MOVING_AVE_DECAY       = 0.9995
__C.YOLO.STRIDES                = [8, 16, 32] #总下采样倍数（Total Downsampling Factor）。在YOLO的语境中，通常也直接称之为步长（Strides）。
__C.YOLO.ANCHOR_PER_SCALE       = 3 #__C.YOLO.ANCHOR_PER_SCALE 这个参数定义了在YOLO的每一个预测尺度（scale）上，使用多少个锚点框（Anchor Box）。
__C.YOLO.IOU_LOSS_THRESH        = 0.5
__C.YOLO.UPSAMPLE_METHOD        = "resize"  
__C.YOLO.ISP_FLAG            = args.ISP_FLAG


# Train options
__C.TRAIN                       = edict()

__C.TRAIN.ANNOT_PATH            = args.train_path
__C.TRAIN.BATCH_SIZE            = 6 #batch_size 
__C.TRAIN.INPUT_SIZE            = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608] #输入图片的不同尺寸 每个epoch开始时，都会随机选择其中的一个size进行训练---多尺度训练
# 然后图片的尺度得是32的倍数:在典型的物体检测网络（如YOLOv3, YOLOv4）中，输入图片会经过多次“下采样”（Down-sampling），通常是通过步长为2的卷积层或池化层实现的。一个常见的网络（如Darknet-53）会进行5次下采样，其总的**下采样率（stride）**就是 2^5 = 32。  #__C.TRAIN.INPUT_SIZE            = [320, 352, 384, 416, 448]

# __C.TRAIN.INPUT_SIZE            = [512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896]
__C.TRAIN.DATA_AUG              = True #有数据增强
__C.TRAIN.LEARN_RATE_INIT       = 1e-6 #学习率为什么有两个值?
__C.TRAIN.LEARN_RATE_END        = 1e-7
__C.TRAIN.WARMUP_EPOCHS         = 0 
__C.TRAIN.FISRT_STAGE_EPOCHS    = args.epoch_first_stage 
__C.TRAIN.SECOND_STAGE_EPOCHS   = args.epoch_second_stage 
__C.TRAIN.INITIAL_WEIGHT        = args.pre_train 



# TEST options
__C.TEST                        = edict()

__C.TEST.ANNOT_PATH             = args.val_path
__C.TEST.BATCH_SIZE             = 6
__C.TEST.INPUT_SIZE             = 544
__C.TEST.DATA_AUG               = False
__C.TEST.WRITE_IMAGE            = True
__C.TEST.WRITE_IMAGE_PATH       = args.WRITE_IMAGE_PATH
__C.TEST.WRITE_IMAGE_SHOW_LABEL = True
__C.TEST.WEIGHT_FILE            = args.WEIGHT_FILE
__C.TEST.SHOW_LABEL             = True
__C.TEST.SCORE_THRESHOLD        = 0.4
__C.TEST.IOU_THRESHOLD          = 0.45

print("--- 评估脚本运行结束！ ---")






