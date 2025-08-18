#! /usr/bin/env python
# coding=utf-8


import os
import time
import shutil
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import core.utils as utils
from tqdm import tqdm #和进度条有关
from core.dataset import Dataset
from core.yolov3 import YOLOV3
from core.config import cfg
from core.config import args
import random
import cv2
import math

from filters import *
print("--- [1] 模块导入成功 ---")

print("gpu型号",args.use_gpu)
if args.use_gpu == 0:
    gpu_id = '-1'
else:
    gpu_id = args.gpu_id
    gpu_list = list()
    gpu_ids = gpu_id.split(',')
    for i in range(len(gpu_ids)):
        gpu_list.append('/gpu:%d' % int(i))
    print(gpu_list)    
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
print("--- [2] GPU环境设置成功 ---")

exp_folder = os.path.join(args.exp_dir, 'exp_{}'.format(args.exp_num))

#导入ckpr检查点 其实实导入概论epoch的模型
set_ckpt_dir = args.ckpt_dir
args.ckpt_dir = os.path.join(exp_folder, set_ckpt_dir)
if not os.path.exists(args.ckpt_dir):
    os.makedirs(args.ckpt_dir)

config_log = os.path.join(exp_folder, 'config.txt')
arg_dict = args.__dict__
msg = ['{}: {}\n'.format(k, v) for k, v in arg_dict.items()]
utils.write_mes(msg, config_log, mode='w')
print("--- [3] 配置文件日志写入成功 ---")





class YoloTrain(object):
    def __init__(self):
        print("--- [4] YoloTrain 初始化开始 (__init__)... ---")
        self.anchor_per_scale    = cfg.YOLO.ANCHOR_PER_SCALE #每个尺度用的锚点框数量
        self.classes             = utils.read_class_names(cfg.YOLO.CLASSES)  #读取总的类别数量
        self.num_classes         = len(self.classes) #5类
        self.learn_rate_init     = cfg.TRAIN.LEARN_RATE_INIT
        self.learn_rate_end      = cfg.TRAIN.LEARN_RATE_END
        self.first_stage_epochs  = cfg.TRAIN.FISRT_STAGE_EPOCHS
        self.second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
        self.warmup_periods      = cfg.TRAIN.WARMUP_EPOCHS
        self.initial_weight      = cfg.TRAIN.INITIAL_WEIGHT
        self.time                = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        self.moving_ave_decay    = cfg.YOLO.MOVING_AVE_DECAY
        self.max_bbox_per_scale  = 150 #每个尺度最多使用候选框的数量150
        self.train_logdir        = "./data/cityfog/log/train"
        self.best_val_loss = np.inf
        print("--- [5] 准备初始化训练集 Dataset('train')... ---")
        
        
        
        
        self.trainset            = Dataset('train') #“我（YoloTrain这个对象）根据 Dataset 的设计蓝图，创建（实例化）了一个专门负责处理训练 ('train') 数据的厨师对象，并将这个新创建的厨师对象，分配给我自己，命名为 self.trainset。” #重点
        print("--- [6] 训练集初始化成功！---")
        print("--- [7] 准备初始化验证集 Dataset('test')... ---")
        self.testset             = Dataset('test') #重点 train 和 dataset的关联 
        print("--- [8] 验证集初始化成功！---")
        self.steps_per_period    = len(self.trainset)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        # self.sess                = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

       
        with tf.name_scope('define_input'): #这行代码的作用是打包。它就像你拿了一个大标签贴在一个区域，上面写着“数据输入区”。这样一来，你定义的所有“投料口”（占位符）都被整齐#  地归纳到了这个区域里，方便管理和后续查看
            self.input_data   = tf.placeholder(tf.float32, [None, None, None, 3], name='input_data') #输入的数据---有雾
            self.defog_A   = tf.placeholder(tf.float32, [None, 3], name='defog_A')
            self.IcA   = tf.placeholder(tf.float32, [None, None, None,1], name='IcA')
            
            #标签/答案数据
            self.label_sbbox  = tf.placeholder(dtype=tf.float32, name='label_sbbox')  #s 代表小尺度物体真实的标签，m代表中，l代表大
            self.label_mbbox  = tf.placeholder(dtype=tf.float32, name='label_mbbox')
            self.label_lbbox  = tf.placeholder(dtype=tf.float32, name='label_lbbox')
            self.true_sbboxes = tf.placeholder(dtype=tf.float32, name='sbboxes') #代表不同尺度物体真实的边界框信息
            self.true_mbboxes = tf.placeholder(dtype=tf.float32, name='mbboxes')
            self.true_lbboxes = tf.placeholder(dtype=tf.float32, name='lbboxes')
            self.input_data_clean   = tf.placeholder(tf.float32, [None, None, None, 3], name='input_data') #输入的数据 ---无雾
            self.trainable     = tf.placeholder(dtype=tf.bool, name='training')

        print("--- [10] TensorFlow 计算图占位符 (placeholder) 定义成功 ---")

        
        
        with tf.name_scope("define_loss"): #计算损失函数并定义模型--yolov3 训练脚本和yolov3脚本的联系
            self.model = YOLOV3(self.input_data, self.trainable, self.input_data_clean, self.defog_A, self.IcA) #修改模型的入口
            t_variables = tf.trainable_variables()#可训练变量
            print("t_variables", t_variables)
            # self.net_var = [v for v in t_variables if not 'extract_parameters' in v.name]
            self.net_var = tf.global_variables() #所有变量
           
            self.giou_loss, self.conf_loss, self.prob_loss, self.recovery_loss = self.model.compute_loss(
                                                    self.label_sbbox,  self.label_mbbox,  self.label_lbbox,
                                                    self.true_sbboxes, self.true_mbboxes, self.true_lbboxes)
            # self.loss only includes the detection loss.
            self.loss = self.giou_loss + self.conf_loss + self.prob_loss + 0.0 * self.recovery_loss
        print("--- [11] 损失函数定义成功 ---")

        
        with tf.name_scope('learn_rate'):
            self.global_step = tf.Variable(1.0, dtype=tf.float64, trainable=False, name='global_step')
            warmup_steps = tf.constant(self.warmup_periods * self.steps_per_period,
                                        dtype=tf.float64, name='warmup_steps')
            train_steps = tf.constant( (self.first_stage_epochs + self.second_stage_epochs)* self.steps_per_period,
                                        dtype=tf.float64, name='train_steps')
            self.learn_rate = tf.cond(
                pred=self.global_step < warmup_steps,
                true_fn=lambda: self.global_step / warmup_steps * self.learn_rate_init,
                false_fn=lambda: self.learn_rate_end + 0.5 * (self.learn_rate_init - self.learn_rate_end) *
                                    (1 + tf.cos(
                                        (self.global_step - warmup_steps) / (train_steps - warmup_steps) * np.pi))
            )
            global_step_update = tf.assign_add(self.global_step, 1.0)

        with tf.name_scope("define_weight_decay"):
            moving_ave = tf.train.ExponentialMovingAverage(self.moving_ave_decay).apply(tf.trainable_variables())
        
        
        
        #第一阶段训练: 先冻结主干 给yolo的训练头热热身 让头部能够更加准确的预测结果
        #您看到的 gamma 和 beta 是**批量归一化（Batch Normalization）**层的可训练参数，它们属于 darknet 主干网络的一部分，所以被过滤掉了。最终列表里只剩下 conv_sbbox,  conv_mbbox, conv_lbbox 的 weight（权重）和 bias（偏置），是因为这段代码的设计目的，就是为了在第一阶段训练中，精确地、只找出这三个预测头的参数来进行更新。
        with tf.name_scope("define_first_stage_train"):
            self.first_stage_trainable_var_list = []
            print("="*30)
            print(">> 开始检查所有可训练变量，以确定第一阶段训练列表...")
                
            for var in tf.trainable_variables():
                var_name = var.op.name
                var_name_mess = str(var_name).split('/')
                 # 打印出当前正在检查的变量名
                print(f"  [*] 正在检查: {var_name}")
                if var_name_mess[0] in ['conv_sbbox', 'conv_mbbox', 'conv_lbbox']:
                     self.first_stage_trainable_var_list.append(var)
                        
                     
            print("\n>> 第一阶段最终训练的变量列表为:")
            for v in self.first_stage_trainable_var_list:
                print(f"  - {v.op.name}")
            print("="*30)        

            #下面逻辑是优化逻辑
            first_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss,
                                                      var_list=self.first_stage_trainable_var_list) #优化器
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([first_stage_optimizer, global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_frozen_variables = tf.no_op()
        
        
        #第二阶段训练: 主干 + 第一阶段的所有可训练变量都要训练 又是adam优化器优化的
        with tf.name_scope("define_second_stage_train"):
            second_stage_trainable_var_list = tf.trainable_variables()
            second_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss,
                                                      var_list=second_stage_trainable_var_list)


            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([second_stage_optimizer, global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_all_variables = tf.no_op()
        print("--- [12] 优化器和训练操作定义成功 ---")


        with tf.name_scope('loader_and_saver'):
            self.loader = tf.train.Saver(self.net_var)
            self.saver  = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        print("--- [13] 模型加载/保存器 (Saver) 定义成功 ---")

       
        #最后每个部分损失算完了， 记录结果和图到logdir中
        with tf.name_scope('summary'):
            tf.summary.scalar("learn_rate",      self.learn_rate)
            tf.summary.scalar("recovery_loss",  self.recovery_loss)
            tf.summary.scalar("giou_loss",  self.giou_loss)
            tf.summary.scalar("conf_loss",  self.conf_loss)
            tf.summary.scalar("prob_loss",  self.prob_loss)
            tf.summary.scalar("total_loss", self.loss)

            # logdir = "./data/log/"
            logdir = os.path.join(exp_folder, 'log')

            if os.path.exists(logdir): shutil.rmtree(logdir)
            os.mkdir(logdir)
            self.write_op = tf.summary.merge_all()
            self.summary_writer  = tf.summary.FileWriter(logdir, graph=self.sess.graph)
        print("--- [14] 日志记录器 (SummaryWriter) 初始化成功 ---")

        
     
    #这里开始才到了train函数 前面那些函数都是初始化。  
    def train(self):
        
        print("--- [15] 进入 train 函数... ---")
        self.sess.run(tf.global_variables_initializer()) #开始正式训练的标志
        print("--- [16] TensorFlow 全局变量初始化成功 ---")
        try:
            print('=> Restoring weights from: %s ... ' % self.initial_weight)
            self.loader.restore(self.sess, self.initial_weight)
        except:
            print('=> %s does not exist !!!' % self.initial_weight)
            print('=> Now it starts to train YOLOV3 from scratch ...')
            self.first_stage_epochs = 0

       
        #这三段是去雾的代码: 一张有雾的照片 = 一张清晰的照片 + 一层半透明的“白纱”（雾）
        #简单说： 这个函数就是在找每个像素“最不受光污染”的底色
        def DarkChannel(im):
            b, g, r = cv2.split(im)
            dc = cv2.min(cv2.min(r, g), b);
            return dc
        
        #简单说： 这个函数在暗通道图里找到了最亮的0.1%的像素点，然后去原图中相同的位置，把这些点的颜色取个平均值，以此作为雾的颜色。
        def AtmLight(im, dark):
            [h, w] = im.shape[:2]
            imsz = h * w
            numpx = int(max(math.floor(imsz / 1000), 1))
            darkvec = dark.reshape(imsz, 1)
            imvec = im.reshape(imsz, 3)

            indices = darkvec.argsort(0)
            indices = indices[(imsz - numpx):imsz]

            atmsum = np.zeros([1, 3])
            for ind in range(1, numpx):
                atmsum = atmsum + imvec[indices[ind]]

            A = atmsum / numpx
            return A
        #简单说： 先用估算出的“雾的颜色A”给原图“洗个澡”，尝试把雾的颜色影响去掉，然后再对“洗完澡”的图片求一次暗通道。这个结果是计算雾有多厚的关键一步。
        def DarkIcA(im, A):
            im3 = np.empty(im.shape, im.dtype)
            for ind in range(0, 3):
                im3[:, :, ind] = im[:, :, ind] / A[0, ind]
            return DarkChannel(im3)

        print("--- [17] 模型权重加载/初始化完成,准备进入Epoch循环... ---")
        
        
 
        #所以，这个 if 语句非常有意义。它是一个开关，一个灵活的选项，允许用户在启动训练时，根据自己的需求，自由选择是否需要以及需要多少轮的第一阶段“热身”训练
        for epoch in range(1, 1+self.first_stage_epochs+self.second_stage_epochs):
            if epoch <= self.first_stage_epochs:
                train_op = self.train_op_with_frozen_variables #第一阶段热身训练
            else:
                train_op = self.train_op_with_all_variables #第二阶段正式训练

            pbar = tqdm(self.trainset)
            print(f"--- [19] Epoch {epoch}: 进度条创建成功，准备遍历数据... (这是最可能出问题的步骤) ---")
            
            train_epoch_loss, test_epoch_loss = [], []


            for train_data in pbar:
                
                #它将两个不同的任务——“图像去雾”和“目标检测”——融合在了一起。 
                #您只需要记住：“哦，原来作者在这里是想让模型同时学习目标检测和图像去雾两个任务，所以准备了很多额外的数据（defog_A, IcA, input_data_clean）喂给它”。
                if args.fog_FLAG:
                    # start_time = time.time()
                    dark = np.zeros((train_data[0].shape[0], train_data[0].shape[1], train_data[0].shape[2]))
                    defog_A = np.zeros((train_data[0].shape[0], train_data[0].shape[3]))
                    IcA = np.zeros((train_data[0].shape[0], train_data[0].shape[1], train_data[0].shape[2]))
                    if DefogFilter in cfg.filters:
                        # print("**************************")
                        for i in range(train_data[0].shape[0]):
                            dark_i = DarkChannel(train_data[0][i])
                            defog_A_i = AtmLight(train_data[0][i], dark_i)
                            IcA_i = DarkIcA(train_data[0][i], defog_A_i)
                            dark[i, ...] = dark_i
                            defog_A[i, ...] = defog_A_i
                            IcA[i, ...] = IcA_i

                    IcA = np.expand_dims(IcA, axis=-1)

                       #train_step_loss 目标检测损失函数  train_step_loss_recovery 图像恢复损失函数
                    _, summary, train_step_loss, global_step_val = self.sess.run(
                        [train_op, self.write_op, self.loss, self.global_step], feed_dict={
                            self.input_data: train_data[0],
                            self.defog_A: defog_A,
                            self.IcA: IcA,
                            self.label_sbbox: train_data[1],
                            self.label_mbbox: train_data[2],
                            self.label_lbbox: train_data[3],
                            self.true_sbboxes: train_data[4],
                            self.true_mbboxes: train_data[5],
                            self.true_lbboxes: train_data[6],
                            self.input_data_clean: train_data[7],
                            self.trainable: True,
                        })


                else: #这段代码就只有目标检测功能
                    _, summary, train_step_loss, global_step_val = self.sess.run(
                        [train_op, self.write_op, self.loss, self.global_step], feed_dict={
                            self.input_data: train_data[7],
                            self.label_sbbox: train_data[1],
                            self.label_mbbox: train_data[2],
                            self.label_lbbox: train_data[3],
                            self.true_sbboxes: train_data[4],
                            self.true_mbboxes: train_data[5],
                            self.true_lbboxes: train_data[6],
                            self.input_data_clean: train_data[7],
                            self.trainable: True,
                        })
                train_epoch_loss.append(train_step_loss)
                self.summary_writer.add_summary(summary, global_step_val)

                pbar.set_description("train loss: %.2f" % train_step_loss) #在进度条上显示“即时”的损失，用于“实时反馈”

            #逻辑和train的一样    
            for batch_idx, test_data in enumerate(tqdm(self.testset, desc=f"Epoch {epoch} - Validating")):
                if args.fog_FLAG:
                # ... (准备 defog_A, IcA 等的代码保持不变, 您可以从您的文件中复制)
                    dark = np.zeros((test_data[0].shape[0], test_data[0].shape[1], test_data[0].shape[2]))
                    defog_A = np.zeros((test_data[0].shape[0], test_data[0].shape[3]))
                    IcA = np.zeros((test_data[0].shape[0], test_data[0].shape[1], test_data[0].shape[2]))
                    if DefogFilter in cfg.filters:
                        for i in range(test_data[0].shape[0]):
                            dark_i = DarkChannel(test_data[0][i])
                            defog_A_i = AtmLight(test_data[0][i], dark_i)
                            IcA_i = DarkIcA(test_data[0][i], defog_A_i)
                            dark[i, ...] = dark_i
                            defog_A[i, ...] = defog_A_i
                            IcA[i, ...] = IcA_i
                    IcA = np.expand_dims(IcA, axis=-1)

                    test_step_loss = self.sess.run(self.loss, feed_dict={
                        self.input_data: test_data[0],
                        self.defog_A: defog_A,
                        self.IcA: IcA,
                        self.label_sbbox: test_data[1],
                        self.label_mbbox: test_data[2],
                        self.label_lbbox: test_data[3],
                        self.true_sbboxes: test_data[4],
                        self.true_mbboxes: test_data[5],
                        self.true_lbboxes: test_data[6],
                        self.input_data_clean: test_data[7],
                        self.trainable: False,
                    })
                else:
                    test_step_loss = self.sess.run(self.loss, feed_dict={
                        self.input_data: test_data[7],
                        self.label_sbbox: test_data[1],
                        self.label_mbbox: test_data[2],
                        self.label_lbbox: test_data[3],
                        self.true_sbboxes: test_data[4],
                        self.true_mbboxes: test_data[5],
                        self.true_lbboxes: test_data[6],
                        self.input_data_clean: test_data[7],
                        self.trainable: False,
                    })

            # ==================== NaN 侦测器 ====================
                if np.isnan(test_step_loss):
                    print(f"\n\n[!!!] 侦测到 NaN Loss！问题可能出在以下一个或多个数据源中 (批次索引: {batch_idx}):")
                    current_batch_size = test_data[0].shape[0]
                    for item_idx in range(current_batch_size):
                        original_index = batch_idx * self.testset.batch_size + item_idx
                        if original_index < self.testset.num_samples:
                            problematic_annotation = self.testset.annotations[original_index]
                            print(f"    -> {problematic_annotation}")
            # =======================================================
            
            test_epoch_loss.append(test_step_loss)
            train_epoch_loss, test_epoch_loss = np.mean(train_epoch_loss), np.mean(test_epoch_loss)  
            #train_epoch_loss, test_epoch_loss = np.mean(train_epoch_loss), np.mean(test_epoch_loss)
            #ckpt_file = args.ckpt_dir + "/yolov3_test_loss=%.4f.ckpt" % test_epoch_loss
            val_summary = tf.Summary()
            val_summary.value.add(tag='Loss/validation_loss', simple_value=test_epoch_loss)
            # 使用训练集的 summary_writer 将其写入，global_step_val 来自训练循环的最后一步
            self.summary_writer.add_summary(val_summary, global_step_val)

            log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            # a. 构建文件名
            ckpt_file = os.path.join(args.ckpt_dir, f"yolov3_test_loss={test_epoch_loss:.4f}.ckpt")

            # b. 打印日志，包含保存信息
            print(f"=> Epoch: {epoch:2d} Time: {log_time} Train loss: {train_epoch_loss:.2f} Test loss: {test_epoch_loss:.2f} Saving {ckpt_file} ...")
        
            # c. 无条件地执行保存
            self.saver.save(self.sess, ckpt_file, global_step=epoch)
#print("--- [20] 所有Epoch训练完成！ ---")


if __name__ == '__main__': 
    print("--- [21] 进入主程序入口 ---")
    YoloTrain().train() # 等效于: m = yolotrain(), m.train(). 
    print("--- [22] 训练任务结束！ ---")



