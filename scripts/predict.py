import os
from ultralytics import YOLO
import cv2

if __name__ == '__main__':

    # model = YOLO('sar_ship_detection/yolov8s_sar_ship/weights/best.pt')
    model = YOLO('runs/detect/train/weights/best.pt')  # 使用训练好的模型
    test_image_dir = 'datasets/dataset/images/test'
    output_dir = 'inference_results'
    os.makedirs(output_dir, exist_ok=True)

    for img_file in os.listdir(test_image_dir):
        if img_file.endswith(('.jpg', '.jpeg')):
            img_path = os.path.join(test_image_dir, img_file)
            results = model.predict(
                source=img_path,
                imgsz=640,
                conf=0.5,  # 置信度阈值
                iou=0.45,
                device=0,
                save=True,
                save_dir=output_dir,
                show=False,
                save_crop=False
            )
            # 在图像上绘制结果
            img = cv2.imread(img_path)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    cls = box.cls[0]
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f"ship: {conf:.2f}", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            save_path = os.path.join(output_dir, img_file)
            cv2.imwrite(save_path, img)
            print(f"处理完成: {img_file}")