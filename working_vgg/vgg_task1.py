import sys
sys.path.append('/home/wangsiyuan/ppq/ppq/samples/TensorRT')
import trt_infer
import os
import torch
from torchvision import models
from PIL import Image
from tqdm import tqdm
from typing import Iterable, List, Tuple
import numpy as np
from ppq import * 
import tensorrt as trt     
import pycuda.driver as cuda
from concurrent.futures import ThreadPoolExecutor

def tensorrt_inference(engine, samples, batch_size: int = 1):
    """ Run a tensorrt model with given samples """
    logger = trt.Logger(trt.Logger.ERROR)
    with open(engine, 'rb') as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    TensorRT_Results = []
    with engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = trt_infer.allocate_buffers(context.engine)
        inputs[0].host = convert_any_to_numpy(samples).astype(np.float32)
        result = trt_infer.do_inference_v2(
            context, bindings=bindings, inputs=inputs, 
            outputs=outputs, stream=stream)[0]
        argmax_result = torch.argmax(convert_any_to_torch_tensor(result).view(-1, 1000), dim=-1)
    return argmax_result

def custom_dataloader(data_dir: str, val_file: str, batch_size: int) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Custom dataloader to load data and labels for evaluation. Automatically detects file type (.npy or .jpg).

    Args:
        data_dir (str): Directory containing the image or data files.
        val_file (str): Path to the validation label file.
        batch_size (int): Number of samples per batch.

    Returns:
        Iterable[Tuple[torch.Tensor, torch.Tensor]]: Batches of (data, labels).
    """


    # Load labels
    with open(val_file, 'r') as f:
        label_map = {os.path.splitext(line.split()[0])[0]: int(line.split()[1]) for line in f}

    # Collect data paths
    data_files = [
        os.path.join(data_dir, file) for file in os.listdir(data_dir)
        if file.endswith(('.npy', '.JPEG'))
    ]
    data_file_names = {os.path.splitext(os.path.basename(file))[0]: file for file in data_files}

    # Filter and match labels and files
    filtered_label_map = {name: label for name, label in label_map.items() if name in data_file_names}
    matched_files = [data_file_names[name] for name in filtered_label_map]
    matched_labels = [filtered_label_map[name] for name in filtered_label_map]

    # Create batches
    def load_file(file):
        if file.endswith('.npy'):
            return torch.tensor(np.load(file), dtype=torch.float32)
        elif file.endswith('.JPEG'):
            return torch.tensor(np.array(Image.open(file).convert('RGB')), dtype=torch.float32)

    for i in range(0, len(matched_files), batch_size):
        batch_files = matched_files[i:i + batch_size]
        batch_labels = matched_labels[i:i + batch_size]

        with ThreadPoolExecutor() as executor:
            data_batch = list(executor.map(load_file, batch_files))

        yield torch.stack(data_batch), torch.tensor(batch_labels, dtype=torch.long)


def evaluate_quantized_engine_model_trt(engine, data_dir: str, val_file: str, batch_size: int, device: str = 'cuda') -> float:
    """
    使用 TensorRT 引擎评估量化模型，数据来自指定目录和标签文件。

    Args:
        engine: TensorRT 引擎文件路径。
        data_dir (str): 数据文件路径。
        val_file (str): 标签文件路径。
        batch_size (int): 批大小。
        device (str): 执行设备 ('cuda' 或 'cpu')。

    Returns:
        float: 模型的分类准确率。
    """
    correct = 0
    total = 0

    # 创建数据加载器
    dataloader = custom_dataloader(data_dir, val_file, batch_size)

    # 批次输入 TensorRT 进行推理并计算准确率
    for data_batch, labels in tqdm(dataloader, desc="Evaluating", unit="batch"):
        # 使用 TensorRT 推理获取结果
        batch_results = tensorrt_inference(engine, data_batch, batch_size=batch_size)

        # 计算预测值并比较
        correct += (batch_results == labels).sum().item()
        total += labels.size(0)

    # 计算准确率
    accuracy = correct / total * 100
    return accuracy
