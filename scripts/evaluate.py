from ultralytics import YOLO

if __name__ == '__main__':
    # model = YOLO('sar_ship_detection/yolov8s_sar_ship/weights/best.pt')
    model = YOLO('runs/detect/train/weights/best.pt')  # 使用训练好的模型

    # 评估模型
    metrics = model.val(
        data='data1.yaml',
        split='test',  # 在测试集上评估
        imgsz=640,
        batch=8,
        device=0,
        verbose=True
    )

    # 打印评估指标
    print(f"mAP@0.5: {metrics.box.map:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map50-95:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")