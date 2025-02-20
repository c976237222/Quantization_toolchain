import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import sys
sys.path.append("/home/wangsiyuan/ppq")
from working_yolo.utils.utils import resize_image, cvtColor, preprocess_input

def convert_images_to_npy(input_directory, output_directory, input_shape=(640, 640), limit=None, letterbox_image=True):
    """
    将输入目录中的图像（.jpg、.jpeg、.png 等）处理并保存为 .npy 文件
    预处理操作与 YOLO 数据预处理流程类似：
    1) 转换为 RGB
    2) 使用灰条填充，保持图像的宽高比
    3) 转为 (C,H,W) 形状的 PyTorch 张量，并将像素值归一化到 [0, 1] 范围
    """
    os.makedirs(output_directory, exist_ok=True)

    # 收集图像文件
    image_files = [f for f in os.listdir(input_directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # 如果指定了 limit，则只处理前 limit 个文件
    if limit is not None:
        image_files = image_files[:limit]

    # 目标输入大小
    target_width, target_height = input_shape

    # 批量处理并保存
    for image_file in tqdm(image_files, desc="Converting images", unit="file"):
        input_path = os.path.join(input_directory, image_file)
        output_path = os.path.join(output_directory, os.path.splitext(image_file)[0] + ".npy")

        try:
            # 打开图片并转换为 RGB
            img = Image.open(input_path).convert("RGB")
            iw, ih = img.size  # 获取图像原始宽高
            w, h = target_width, target_height

            # 使用 resize_image 函数进行灰条填充，保持宽高比
            img = resize_image(img, (w, h), letterbox_image=letterbox_image)

            # 转为 NumPy 数组，并归一化
            img_data = np.array(img, dtype=np.float32)
            img_data /= 255.0  # 归一化到 [0, 1]

            # 转换成 (C, H, W) 格式
            img_data = np.transpose(img_data, (2, 0, 1))  # 转换为 (C, H, W)

            # 预处理
            #img_data = preprocess_input(img_data)

            # 保存为 .npy 文件
            np.save(output_path, img_data)

            print(f"Processed {image_file}: shape {img_data.shape}, dtype {img_data.dtype}, "
                  f"min={img_data.min():.3f}, max={img_data.max():.3f}")

        except Exception as e:
            print(f"Error processing {image_file}: {e}")


# 示例用法
if __name__ == "__main__":
    input_directory = "/home/wangsiyuan/ssd-pytorch/VOCdevkit_1/VOC2007/JPEGImages"  # 输入图像目录
    output_directory = "/home/wangsiyuan/ppq/working_ssd/data"  # 输出 .npy 文件的目录
    convert_images_to_npy(input_directory, output_directory, input_shape=(300, 300), limit=1000)
