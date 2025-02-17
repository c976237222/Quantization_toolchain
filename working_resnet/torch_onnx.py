import torch
from torchvision.models import resnet18

# Step 1: 加载预训练模型
model = resnet18(pretrained=True)
model.eval()  # 设置为评估模式

# Step 2: 创建一个示例输入
# ResNet-18 的输入大小为 [batch_size, 3, 224, 224]，其中 3 是 RGB 通道数
dummy_input = torch.randn(1, 3, 224, 224)

# Step 3: 导出为 ONNX 模型
onnx_file_path = "/home/wangsiyuan/ppq/working_resnet/resnet18.onnx"
torch.onnx.export(
    model,                          # PyTorch 模型
    dummy_input,                    # 示例输入
    onnx_file_path,                 # 导出的 ONNX 文件路径
    export_params=True,             # 导出训练好的参数权重
    opset_version=13,               # ONNX opset 版本
    do_constant_folding=True,      # 是否进行常量折叠优化
    input_names=["input"],          # 输入节点名称
    output_names=["output"],        # 输出节点名称
    dynamic_axes={                  # 动态轴支持（可选）
        "input": {0: "batch_size"}, # 批量大小可以动态变化
        "output": {0: "batch_size"}
    }
)

print(f"Model exported to {onnx_file_path}")
