import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm


def convert_images_to_npy(input_directory, output_directory, input_shape=(300, 300), limit=None):
    """
    将输入目录中的图像（.jpg、.jpeg、.png 等）处理并保存为 .npy 文件
    预处理操作类似于 SSD 验证集时的处理：
    1) 按比例缩放到目标大小
    2) 用灰色填充，保持图像的宽高比
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

            # 计算缩放比例，保持宽高比
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)

            # 缩放图像
            img = img.resize((nw, nh), Image.BICUBIC)

            # 创建一个新的灰色背景图片，并将缩放后的图像粘贴到中心
            new_image = Image.new('RGB', (w, h), (128, 128, 128))  # 灰色背景
            new_image.paste(img, ((w - nw) // 2, (h - nh) // 2))

            # 转为 NumPy 数组，并归一化
            img_data = np.array(new_image, dtype=np.float32)
            img_data /= 255.0  # 归一化到 [0, 1]

            # 转换成 (C, H, W) 格式
            img_data = np.transpose(img_data, (2, 0, 1))  # 转换为 (C, H, W)

            # 保存为 .npy 文件
            np.save(output_path, img_data)

            print(f"Processed {image_file}: shape {img_data.shape}, dtype {img_data.dtype}, "
                  f"min={img_data.min():.3f}, max={img_data.max():.3f}")

        except Exception as e:
            print(f"Error processing {image_file}: {e}")


# 示例用法
if __name__ == "__main__":
    input_directory = "/home/wangsiyuan/ssd-pytorch/VOCdevkit_1/VOC2007/JPEGImages"  # 输入图像目录
    output_directory = "/share/wangsiyuan-local/datasets/VOCdevkit/VOC2007/npy_ssd"  # 输出 .npy 文件的目录
    convert_images_to_npy(input_directory, output_directory, input_shape=(300, 300), limit=1000)
