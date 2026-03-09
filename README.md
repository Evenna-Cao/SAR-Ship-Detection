# SAR Ship Detection

This project implements ship detection in SAR (Synthetic Aperture Radar) images using YOLOv8, optimized for RTX5070 GPUs. It involves environment configuration with CUDA 12.8 and PyTorch, TIFF-to-JPG image conversion, XML-to-YOLO annotation format transformation, and dataset splitting (80/10/10 for train/val/test). The YOLOv8s model is trained with Adam optimizer, cosine learning rate scheduling, and early stopping. Post-training, model evaluation (mAP, precision, recall) and inference on test images are conducted, with detection results visualized and saved. Key optimizations include batch size adjustment for RTX5070 memory constraints and data augmentation to mitigate overfitting. The project delivers a complete pipeline from data preprocessing to model deployment for SAR ship detection tasks.

## 1. Environment Configuration

### 1.1 Create Conda Virtual Environment

### 1.2 Install CUDA and PyTorch

### 1.3 Install YOLOv8 and Dependencies
```bash
pip install ultralytics
pip install opencv-python pillow pandas tqdm
```


## 2. Program Execution

```bash
python scripts/convert_tiff_to_jpg.py
python scripts/convert_xml_to_yolo.py
python scripts/split_dataset.py
python scripts/train.py
python scripts/evaluate.py
python scripts/predict.py

```


## 3. Project Structure

`raw_data`：Raw dataset

`datasets`：Processed dataset

`scripts`：Scripts for each step

`runs`：Save model weights and evaluation results

`inference_results`：Save inference results

