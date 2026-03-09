import os
from PIL import Image

source_folder = "raw_data/tiff_images"
destination_folder = "datasets/dataset/images"

os.makedirs(destination_folder, exist_ok=True)

for filename in os.listdir(source_folder):
    if filename.lower().endswith(('.tif', '.tiff')):
        img_path = os.path.join(source_folder, filename)
        try:
            with Image.open(img_path) as img:
                # SAR图像通常是单通道，转换为3通道RGB
                if img.mode != 'RGB':
                    rgb_img = img.convert('RGB')
                else:
                    rgb_img = img

                jpg_filename = os.path.splitext(filename)[0] + '.jpg'
                save_path = os.path.join(destination_folder, jpg_filename)
                rgb_img.save(save_path, 'JPEG', quality=95)
                print(f"转换完成: {filename} -> {jpg_filename}")

        except Exception as e:
            print(f"转换 {filename} 时出错: {e}")



# import os
# from PIL import Image
#
# source_folder = "raw_data/tiff_images"  # 修改为你的TIFF图像路径
# destination_folder = "dataset/images"
#
# os.makedirs(destination_folder, exist_ok=True)
#
# for filename in os.listdir(source_folder):
#     if filename.lower().endswith(('.tif', '.tiff')):
#         img_path = os.path.join(source_folder, filename)
#         with Image.open(img_path) as img:
#             # 转换为RGB模式（处理单通道SAR图像）
#             rgb_img = img.convert('RGB')
#             jpg_filename = os.path.splitext(filename)[0] + '.jpg'
#             save_path = os.path.join(destination_folder, jpg_filename)
#             rgb_img.save(save_path, 'JPEG', quality=95)
#             print(f"转换完成: {filename} -> {jpg_filename}")
