import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from torchvision import transforms

def convert_images_to_npy(input_directory, output_directory, limit=None):
    """
    将输入目录中的图像（.jpg、.jpeg、.png 等）读入后：
      1) 统一缩放到 224x224（ResNet-50 的输入大小）
      2) 转为 (C,H,W) 形状的 PyTorch 张量，数值范围标准化为 ImageNet 格式
      3) 保存为 .npy 文件

    适用于 ResNet-50 等模型，需 ImageNet 标准化 (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])。
    """
    os.makedirs(output_directory, exist_ok=True)

    # 收集图像文件
    image_files = [f for f in os.listdir(input_directory)
                if f.endswith('.JPEG') or f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')]


    # 如果指定了 limit，则只处理前 limit 个文件
    if limit is not None:
        image_files = image_files[:limit]

    # 构建预处理 transforms：
    # - Resize 到 (224,224)
    # - ToTensor() -> [0,1]，形状 (C,H,W)
    # - Normalize(mean, std) -> 标准化到 ImageNet 格式
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet-50 输入大小为 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 标准化
    ])

    # 批量处理并保存
    for image_file in tqdm(image_files, desc="Converting images", unit="file"):
        input_path = os.path.join(input_directory, image_file)
        output_path = os.path.join(output_directory, os.path.splitext(image_file)[0] + ".npy")

        try:
            # 打开图片、转 RGB
            img = Image.open(input_path).convert("RGB")
            # 执行预处理
            img_tensor = transform(img)  # shape: (3, 224, 224)
            # 转 numpy 保存
            img_npy = img_tensor.numpy()
            np.save(output_path, img_npy)
            print(f"Processed {image_file}: shape {img_npy.shape}, dtype {img_npy.dtype}, "
                  f"min={img_npy.min():.3f}, max={img_npy.max():.3f}")

        except Exception as e:
            print(f"Error processing {image_file}: {e}")

# 示例用法
if __name__ == "__main__":
    input_directory = "/share/wangsiyuan-local/datasets/imagenet/val5000_quant"    # COCO val2017 图片目录
    output_directory = "/share/wangsiyuan-local/datasets/imagenet/val5000_npy"    # 输出 .npy 目录
    convert_images_to_npy(input_directory, output_directory, limit=3000)
