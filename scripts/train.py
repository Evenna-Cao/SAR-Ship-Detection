from ultralytics import YOLO
import os
import torch

if __name__ == '__main__':
    # 在Windows上必须添加这个保护
    torch.multiprocessing.freeze_support()

    # 加载模型
    # 使用本地模型文件
    model_path = '../yolov8s.pt'  # 相对于脚本的路径
    if os.path.exists(model_path):
        model = YOLO(model_path)
    else:
        # 如果本地文件不存在，fallback 到在线下载
        model = YOLO('yolov8s.pt')


    # # 训练模型
    # results = model.train(
    #     data='data1.yaml',
    #     epochs=100,
    #     batch=16,
    #     imgsz=640,
    #     device=0,
    #     # workers=4,  # 减少workers数量
    #     project='sar_ship_detection',
    #     name='yolov8s_sar_ship7',
    #     exist_ok=False,
    #     pretrained=True,
    #     verbose=True
    # )

    # 训练模型
    results = model.train(
        data='data1.yaml',
        epochs=50,
        batch=8,  # RTX5070 12GB显存推荐值
        imgsz=640,
        device=0,  # 使用第1块GPU
        optimizer='Adam',  # 适合小数据集的优化器
        lr0=0.01,  # 初始学习率
        weight_decay=0.0005,
        warmup_epochs=3,
        cos_lr=True,  # 余弦学习率调度
        save=True,
        save_period=5,
        patience=10,  # 早停策略
        pretrained=True,
        verbose=True,
        # project='SSD',
        # name='yolov8s_sar_ship'
    )