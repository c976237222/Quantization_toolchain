import torch
from torchvision import models
import onnx

# 步骤 1：加载模型
model = models.vgg16(weights=True)
model.eval()

# 步骤 2：准备示例输入
dummy_input = torch.randn(1, 3, 224, 224)

# 步骤 3：导出为 ONNX
onnx_model_path = "vgg16.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_model_path,
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names = ['input'],
    output_names = ['output'],
    dynamic_axes={'input' : {0 : 'batch_size'},
                  'output' : {0 : 'batch_size'}}
)

print(f"模型已成功导出到 {onnx_model_path}")

# 步骤 4：验证模型
onnx_model = onnx.load(onnx_model_path)
try:
    onnx.checker.check_model(onnx_model)
    print("ONNX 模型是有效的。")
except onnx.checker.ValidationError as e:
    print("ONNX 模型无效：", e)
