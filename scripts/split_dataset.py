import os
import shutil
import random

# 配置
image_dir = "datasets/dataset/images"
label_dir = "datasets/dataset/labels"
output_dir = "dataset"
train_ratio = 0.8
val_ratio = 0.1

# 创建目录结构
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

# 获取所有图像文件
images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg'))]
random.shuffle(images)

# 划分索引
total = len(images)
train_idx = int(total * train_ratio)
val_idx = int(total * (train_ratio + val_ratio))

# 复制文件
for i, img in enumerate(images):
    img_path = os.path.join(image_dir, img)
    label_path = os.path.join(label_dir, os.path.splitext(img)[0] + '.txt')

    if i < train_idx:
        split = 'train'
    elif i < val_idx:
        split = 'val'
    else:
        split = 'test'

    shutil.copy(img_path, os.path.join(output_dir, 'images', split, img))
    if os.path.exists(label_path):
        shutil.copy(label_path, os.path.join(output_dir, 'labels', split, os.path.basename(label_path)))