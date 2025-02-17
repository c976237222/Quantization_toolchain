import torch
import onnx
import argparse
from torchvision import models

# YOLOv5 相关库
def load_yolov5(model_name="yolov8x"):
    from ultralytics import YOLO
    return YOLO(model_name + ".pt")

def export_model_to_onnx(model_name, output_path="model.onnx"):
    if model_name == "vgg16":
        print("加载 VGG16 模型...")
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224)
    elif model_name == "yolov8x":
        print("加载 yolov8x 模型...")
        model = load_yolov5()
        dummy_input = torch.randn(1, 3, 640, 640)  # YOLOv5 需要 640x640 输入
    else:
        raise ValueError("不支持的模型: " + model_name)
    
    print(f"导出 {model_name} 到 {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"{model_name} 已成功导出到 {output_path}")
    
    # 验证 ONNX 模型
    onnx_model = onnx.load(output_path)
    try:
        onnx.checker.check_model(onnx_model)
        print("ONNX 模型是有效的。")
    except onnx.checker.ValidationError as e:
        print("ONNX 模型无效：", e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["vgg16", "yolov8x"], required=True, help="选择要导出的模型")
    parser.add_argument("--output", default="model.onnx", help="导出的 ONNX 文件名")
    args = parser.parse_args()
    
    export_model_to_onnx(args.model, args.output)