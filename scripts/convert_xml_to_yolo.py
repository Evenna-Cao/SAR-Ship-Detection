import os
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm

# 配置
XML_DIR = "raw_data/xml_annotations"  # XML文件路径
IMG_DIR = "datasets/dataset/images"  # 图像路径（转换后的JPG图像）
OUTPUT_DIR = "datasets/dataset/labels"
CLASSES = ["ship"]  # 仅舰船一个类别

os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_bbox_from_points(points_element):
    """
    从多边形点中提取边界框 [x_min, y_min, x_max, y_max]
    示例输入: <points><point>944,986</point><point>944,999</point>...</points>
    """
    points = []
    for point_element in points_element.findall('point'):
        x, y = map(int, point_element.text.split(','))
        points.append((x, y))

    # 提取边界框坐标
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def convert(size, box):
    """将绝对坐标转换为YOLO格式的相对坐标"""
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[2]) / 2.0  # 中心点x
    y = (box[1] + box[3]) / 2.0  # 中心点y
    w = box[2] - box[0]  # 宽度
    h = box[3] - box[1]  # 高度
    return (round(x * dw, 6), round(y * dh, 6), round(w * dw, 6), round(h * dh, 6))


# 处理所有XML文件
for xml_file in tqdm(os.listdir(XML_DIR)):
    if not xml_file.endswith('.xml'):
        continue

    try:
        tree = ET.parse(os.path.join(XML_DIR, xml_file))
        root = tree.getroot()

        # 获取对应的图像文件名（从XML中读取）
        filename_elem = root.find('.//filename')
        if filename_elem is None:
            print(f"警告: {xml_file} 中没有找到filename标签，跳过")
            continue

        # 确保使用.jpg扩展名（因为我们已经转换了图像）
        img_filename = os.path.splitext(filename_elem.text)[0] + '.jpg'
        img_path = os.path.join(IMG_DIR, img_filename)

        # 获取图像尺寸
        if os.path.exists(img_path):
            with Image.open(img_path) as img:
                width, height = img.size
        else:
            print(f"警告: 图像文件 {img_filename} 不存在，跳过 {xml_file}")
            continue

        # 创建对应的YOLO标注文件
        txt_filename = os.path.splitext(xml_file)[0] + '.txt'
        txt_path = os.path.join(OUTPUT_DIR, txt_filename)

        with open(txt_path, 'w') as f:
            # 查找所有的object标签
            for obj in root.findall('.//object'):
                # 获取类别
                name_elem = obj.find('.//name')
                if name_elem is None or name_elem.text not in CLASSES:
                    continue

                cls_id = CLASSES.index(name_elem.text)

                # 从points中提取边界框
                points_elem = obj.find('points')
                if points_elem is None:
                    continue

                x_min, y_min, x_max, y_max = extract_bbox_from_points(points_elem)

                # 转换为YOLO格式
                bbox_yolo = convert((width, height), (x_min, y_min, x_max, y_max))
                f.write(f"{cls_id} {' '.join(map(str, bbox_yolo))}\n")

        print(f"转换完成: {xml_file} -> {txt_filename}")

    except Exception as e:
        print(f"处理 {xml_file} 时出错: {e}")
        continue