import os
import torch
from torchvision import transforms, datasets
from tqdm import tqdm
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import trt_infer  # Assuming trt_infer contains helper functions for TensorRT inference

def prepare_transforms():
    """
    数据预处理操作，包括调整大小、归一化等。
    Returns:
        transforms.Compose: 数据预处理组合
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet-50 输入大小为 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 标准化
    ])

def load_test_data(test_dir, batch_size=100, num_workers=4):
    """
    加载测试数据。

    Args:
        test_dir (str): 测试集目录
        batch_size (int): 批大小
        num_workers (int): DataLoader 的工作线程数

    Returns:
        DataLoader: 测试数据加载器
    """
    transform = prepare_transforms()
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print(f"Test dataset size: {len(test_dataset)}")
    return test_loader

def tensorrt_inference(engine_path, samples, batch_size: int = 1):
    """
    使用 TensorRT 模型进行推理。

    Args:
        engine_path (str): TensorRT 引擎文件路径。
        samples (torch.Tensor): 输入样本。
        batch_size (int): 每批次样本数。

    Returns:
        torch.Tensor: 推理结果的分类索引。
    """
    logger = trt.Logger(trt.Logger.ERROR)
    with open(engine_path, 'rb') as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    with engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = trt_infer.allocate_buffers(context.engine)
        context.set_binding_shape(0, samples.shape) 
        inputs[0].host = samples.cpu().numpy().astype(np.float32)  # Move tensor to CPU before converting to numpy
        result = trt_infer.do_inference(
            context, bindings=bindings, inputs=inputs,
            outputs=outputs, stream=stream, batch_size=batch_size
        )[0]
        argmax_result = torch.argmax(torch.tensor(result).view(-1, 1000), dim=-1).to(samples.device)  # Ensure result is on the same device
    return argmax_result

def evaluate_resnet_tensorrt_engine(engine_path, test_loader, device='cuda'):
    """
    在测试集上评估量化的 ResNet TensorRT 引擎。

    Args:
        engine_path (str): TensorRT 引擎文件路径。
        test_loader (DataLoader): 测试数据加载器。
        device (str): 执行设备 ('cuda' 或 'cpu')。

    Returns:
        float: 模型的分类准确率。
    """
    correct = 0
    total = 0

    for data_batch, labels in tqdm(test_loader, desc="Evaluating", unit="batch"):
        # 将数据移至设备
        data_batch = data_batch.to(device)
        labels = labels.to(device)  # Ensure labels are on the same device

        # 使用 TensorRT 进行推理
        batch_results = tensorrt_inference(engine_path, data_batch, batch_size=len(data_batch))
        # 计算预测准确性
        correct += (batch_results == labels).sum().item()
        print(correct)
        total += labels.size(0)

    accuracy = correct / total * 100
    print(f"Accuracy on test dataset: {accuracy:.2f}%")
    return accuracy

# 示例用法
# engine_path = 'path/to/resnet50.trt'
# test_dir = 'path/to/imagenet/val'
# batch_size = 32
# test_loader = load_test_data(test_dir, batch_size)
# evaluate_resnet_tensorrt_engine(engine_path, test_loader)