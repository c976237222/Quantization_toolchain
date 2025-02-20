from ultralytics import YOLO
import os
import torch

# 检查 CUDA 和环境变量
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")

# 设置 GPU 环境
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 加载预训练模型（推荐用于训练）
model = YOLO("working_yolo/yolov5nu.pt")

# 导出为 ONNX 文件（指定 opset 版本为 13）
success = model.export(
    format="engine",
    half=False,
    int8=False,        # 不使用 FP16 精度
    device='cuda',        # 在 GPU 上导出
    simplify=True,       # 不简化 ONNX 模型
    nms=False,            # 不添加 NMS 操作
    opset=13,              # 指定 ONNX 的 opset 版本为 13
    dynamic=True,
    batch=64
)

print("YOLOv8 模型已成功导出为 ONNX" if success else "导出失败")
