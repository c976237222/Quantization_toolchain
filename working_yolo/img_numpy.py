import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from torchvision import transforms

def convert_images_to_npy(input_directory, output_directory, limit=None):
    """
    将输入目录中的图像（.jpg、.jpeg、.png 等）读入后：
      1) 统一缩放到 640x640
      2) 转为 (C,H,W) 形状的 PyTorch 张量，数值范围 [0,1]
      3) 保存为 .npy 文件

    适用于 YOLOv5/YOLOv8 等需要 (3,640,640) 并接受 [0,1] 输入范围的检测模型。
    如若模型还需要 letterbox，请在此处实现相应逻辑。
    """
    os.makedirs(output_directory, exist_ok=True)

    # 收集图像文件
    image_files = [f for f in os.listdir(input_directory)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # 如果指定了 limit，则只处理前 limit 个文件
    if limit is not None:
        image_files = image_files[:limit]

    # 构建预处理 transforms：
    # - Resize 到 (640,640)
    # - ToTensor() -> [0,1]，形状 (C,H,W)
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor()
    ])

    # 批量处理并保存
    for image_file in tqdm(image_files, desc="Converting images", unit="file"):
        input_path = os.path.join(input_directory, image_file)
        output_path = os.path.join(output_directory, os.path.splitext(image_file)[0] + ".npy")

        try:
            # 打开图片、转 RGB
            img = Image.open(input_path).convert("RGB")
            # 执行预处理
            img_tensor = transform(img)  # shape: (3, 640, 640), range: [0, 1]
            # 转 numpy 保存
            img_npy = img_tensor.numpy()
            np.save(output_path, img_npy)
            print(f"Processed {image_file}: shape {img_npy.shape}, dtype {img_npy.dtype}, "
                  f"min={img_npy.min():.3f}, max={img_npy.max():.3f}")

        except Exception as e:
            print(f"Error processing {image_file}: {e}")

# 示例用法
if __name__ == "__main__":
    input_directory = "/share/wangsiyuan-local/datasets/coco/images/val2017"    # COCO val2017 图片目录
    output_directory = "/home/wangsiyuan/ppq/working_yolo/data"    # 输出 .npy 目录
    convert_images_to_npy(input_directory, output_directory, limit=640)
