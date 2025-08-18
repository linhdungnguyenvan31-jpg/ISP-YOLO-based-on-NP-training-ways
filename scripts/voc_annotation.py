# 整个脚本的工作流程是这样的：
# 脚本会读取每张图片对应的标注文件（一个.xml文件）。
# 它会遍历这个标注文件里记录的所有物体。
# 对于每一个物体，它会检查这个物体的名字（比如 'aeroplane', 'boat', 'person' 等）是否在你上面代码里那个特定的列表 ['person', 'car', 'bus', 'bicycle', 'motorbike'] 之中。
# 当且仅当物体的名字在这个列表里，脚本才会把它的坐标和类别索引追加到输出的那一行。
# 如果一张图片（比如 009861.jpg）虽然标注了物体，但这些物体都不是“行人、汽车、公交车、自行车或摩托车”中的任何一种，那么 if 条件就永远不会满足。因此，不会有任何数字被追加到那一行，最终你看到的输出就只有图片路径了。


import os
import argparse
import xml.etree.ElementTree as ET

def convert_voc_annotation(data_path, data_type, anno_path, use_difficult_bbox=False):

    # classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    #            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
    #            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
    #            'train', 'tvmonitor']
    classes = ['person', 'car', 'bus', 'bicycle',  'motorbike']
    img_inds_file = os.path.join(data_path, 'ImageSets', 'Main', data_type + '.txt')
    with open(img_inds_file, 'r') as f:
        txt = f.readlines()
        image_inds = [line.strip() for line in txt]

    with open(anno_path, 'a') as f:
        for image_ind in image_inds:
            image_path = os.path.join(data_path, 'JPEGImages', image_ind + '.jpg')
            annotation = image_path
            label_path = os.path.join(data_path, 'Annotations', image_ind + '.xml')
            root = ET.parse(label_path).getroot()
            objects = root.findall('object')
            for obj in objects:
                difficult = obj.find('difficult').text.strip()
                if (not use_difficult_bbox) and(int(difficult) == 1):
                    continue
                bbox = obj.find('bndbox')
                if obj.find('name').text.lower().strip() in ['person', 'car', 'bus', 'bicycle',  'motorbike']:
                    class_ind = classes.index(obj.find('name').text.lower().strip())
                    xmin = bbox.find('xmin').text.strip()
                    xmax = bbox.find('xmax').text.strip()
                    ymin = bbox.find('ymin').text.strip()
                    ymax = bbox.find('ymax').text.strip()
                    annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_ind)])

            print(annotation)
            f.write(annotation + "\n")
    return len(image_inds)

if __name__ == '__main__':
    # for foggy conditions
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./data/VOC/",help="path to VOC dataset")
    parser.add_argument("--train_annotation", default="./data/dataset_fog/voc_norm_train.txt")
    parser.add_argument("--test_annotation",  default="./data/dataset_fog/voc_norm_test.txt")

    flags = parser.parse_args()

    if os.path.exists(flags.train_annotation):os.remove(flags.train_annotation)
    #if os.path.exists(flags.val_annotation):os.remove(flags.val_annotation)
    if os.path.exists(flags.test_annotation):os.remove(flags.test_annotation)

    num1 = convert_voc_annotation(os.path.join(flags.data_path, 'VOC2007'), 'trainval', flags.train_annotation, False)
    num2 = convert_voc_annotation(os.path.join(flags.data_path, 'VOC2012'), 'trainval', flags.train_annotation, False)
    num3 = convert_voc_annotation(os.path.join(flags.data_path, 'VOC2007'),  'test', flags.test_annotation, False)
    print('=> The number of image for train is: %d\tThe number of image for train is:%d\tThe number of image for test is:%d' %(num1, num2, num3))



