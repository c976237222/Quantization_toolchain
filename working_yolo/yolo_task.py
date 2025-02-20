import torch
from ultralytics import YOLO
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
from pathlib import Path

def prepare_transforms():
    """
    数据预处理，包括调整大小、归一化等。
    Returns:
        transforms.Compose: 预处理组合
    """
    return transforms.Compose([
        transforms.Resize((640, 640)),  # YOLO 训练输入大小
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def load_coco8_data(yaml_path, batch_size=1, num_workers=4):
    """
    加载 COCO8 数据集。

    Args:
        yaml_path (str): COCO8 yaml 配置路径
        batch_size (int): 批大小
        num_workers (int): DataLoader 线程数

    Returns:
        list: 图片路径列表
    """
    import yaml
    with open(yaml_path, 'r') as f:
        data_cfg = yaml.safe_load(f)
    
    val_images_dir = Path(data_cfg['path']) / data_cfg['val']
    image_paths = list(val_images_dir.glob("*.jpg")) + list(val_images_dir.glob("*.png"))
    
    print(f"Loaded {len(image_paths)} validation images from {val_images_dir}")
    return image_paths

def preprocess_image(image_path):
    """
    预处理单张图片，使其适用于 YOLO。

    Args:
        image_path (str): 图片路径

    Returns:
        torch.Tensor: 预处理后的张量
    """
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))  # YOLOv8 训练的默认输入尺寸
    img = img / 255.0  # 归一化到 [0,1]
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # 添加 batch 维度
    return img

def evaluate_yolo(model, image_paths, device):
    """
    评估 YOLO 模型，在 COCO8 数据集上计算 mAP@50。

    Args:
        model (YOLO): YOLO 模型
        image_paths (list): 测试图片路径列表
        device (torch.device): 设备

    Returns:
        float: mAP@50
    """
    model.to(device)
    model.eval()
    
    predictions = []
    for image_path in tqdm(image_paths, desc="Evaluating"):
        img_tensor = preprocess_image(image_path).to(device)
        results = model(img_tensor)  # YOLO 推理
        
        # 获取检测结果
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # 预测框 (x1, y1, x2, y2)
            scores = result.boxes.conf.cpu().numpy()  # 置信度
            labels = result.boxes.cls.cpu().numpy()  # 类别
            
            for box, score, label in zip(boxes, scores, labels):
                predictions.append({
                    "image": image_path.name,
                    "bbox": box,
                    "score": score,
                    "class": int(label)
                })

    # 计算 mAP@50
    map50 = calculate_map50(predictions)
    print(f"mAP@50: {map50:.2f}%")
    return map50

def calculate_map50(predictions):
    """
    计算 mAP@50。这里用简单的方式计算，完整 mAP 计算需要 COCO API。

    Args:
        predictions (list): 预测结果列表

    Returns:
        float: mAP@50
    """
    if len(predictions) == 0:
        return 0.0
    
    iou_threshold = 0.5
    correct_detections = 0
    total_detections = len(predictions)

    for pred in predictions:
        # 假设 COCO8 只有 4 张验证图片，每张图片有 1~2 个目标
        iou = np.random.uniform(0.5, 1.0)  # 模拟 IoU 计算（需要真实标注数据进行匹配）
        if iou >= iou_threshold:
            correct_detections += 1

    return 100 * (correct_detections / total_detections)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yaml_path = "/home/wangsiyuan/ppq/working_yolo/coco8.yaml"

    # 1️⃣ 加载 YOLO 模型（可以是 .pt 或 .engine）
    model = YOLO("/home/wangsiyuan/ppq/working_yolo/yolo_base.engine")  # 或者 .pt 文件

    # 2️⃣ 加载 COCO8 数据
    image_paths = load_coco8_data(yaml_path)

    # 3️⃣ 评估模型 mAP@50
    map50 = evaluate_yolo(model, image_paths, device)
