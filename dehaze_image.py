import cv2
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os

# --- 导入您项目中的核心模块 ---
import core.utils as utils
from core.yolov3 import YOLOV3
from core.config import cfg

# ==================== 1. 配置区 (请在这里修改) ====================

# a. 指定您训练好的、最佳的模型权重文件路径
WEIGHT_FILE_PATH = './best_models/yolov3_test_loss=2.9701.ckpt-76'

# b. 指定您想要去雾的单张图片路径
INPUT_IMAGE_PATH = './SJZ_Bing_416.png'

# c. 指定保存结果图的文件名
OUTPUT_IMAGE_PATH = './best_models/dehaze_result_comparison.jpg'

# =================================================================

class ImageDehazer(object):
    def __init__(self, weight_path):
        self.input_size = cfg.TEST.INPUT_SIZE
        self.trainable  = tf.placeholder(dtype=tf.bool, name='training')
        self.input_data = tf.placeholder(tf.float32, [1, None, None, 3], name='input_data')
        self.input_data_clean = tf.placeholder(tf.float32, [1, None, None, 3], name='input_data_clean')
        self.defog_A    = tf.placeholder(tf.float32, [1, 3], name='defog_A')
        self.IcA        = tf.placeholder(tf.float32, [1, None, None, 1], name='IcA')

        # 构建模型图
        model = YOLOV3(self.input_data, self.trainable, self.input_data_clean, self.defog_A, self.IcA)
        # 我们只关心图像增强后的输出
        self.enhanced_image_tensor = model.image_isped

        # 加载模型
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        
        print("=> 正在加载模型权重...")
        saver = tf.train.Saver()
        saver.restore(self.sess, weight_path)
        print("=> 模型加载成功！")

    def dehaze(self, image):
        """对单张图片进行去雾处理"""
        original_h, original_w, _ = image.shape
        
        # 预处理：将图片缩放并填充为模型需要的尺寸 (e.g., 416x416)
        image_processed, _ = utils.image_preporcess(np.copy(image), [self.input_size, self.input_size])
        image_data = image_processed[np.newaxis, ...]

        # 运行模型的前向传播，只获取增强后的图片
        # 注意：我们不需要defog_A和IcA，因为在纯推理时，模型可以不依赖它们
        dummy_defog_A = np.zeros((1, 3))
        dummy_IcA = np.zeros((1, self.input_size, self.input_size, 1))

        enhanced_image = self.sess.run(self.enhanced_image_tensor,
                                       feed_dict={
                                           self.input_data: image_data,
                                           self.defog_A: dummy_defog_A,
                                           self.IcA: dummy_IcA,
                                           self.trainable: False
                                       })
        
        # 反处理：将增强后的图片从416x416恢复到原始尺寸
        restored_image, _ = utils.image_unpreporcess(
            image=enhanced_image[0],
            target_size=(original_h, original_w),
            gt_boxes=None
        )
        
        return restored_image

if __name__ == '__main__':
    # 检查输入文件是否存在
    if not os.path.exists(INPUT_IMAGE_PATH):
        print(f"[错误] 输入图片不存在: {INPUT_IMAGE_PATH}")
    else:
        # 加载去雾器
        dehazer = ImageDehazer(WEIGHT_FILE_PATH)
        
        # 读取原始雾图
        original_foggy_image = cv2.imread(INPUT_IMAGE_PATH)
        
        # 执行去雾
        print("=> 正在进行去雾处理...")
        dehazed_image = dehazer.dehaze(original_foggy_image)
        print("=> 去雾处理完成！")
        
        # 创建对比图
        # 为了方便对比，我们将原始雾图也调整到和输出一样的大小
        h, w, _ = dehazed_image.shape
        original_foggy_image_resized = cv2.resize(original_foggy_image, (w, h))
        
        # 添加文字标签
        cv2.putText(original_foggy_image_resized, "Input(Foggy)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(dehazed_image, "ISP-YOLO Output(Dehazed)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # 将两张图片横向拼接在一起
        comparison_image = np.concatenate([original_foggy_image_resized, dehazed_image], axis=1)
        
        # 保存最终的对比图
        cv2.imwrite(OUTPUT_IMAGE_PATH, comparison_image)
        print(f"=> 成功！对比图已保存到: {OUTPUT_IMAGE_PATH}")