import os
import torch
from torchvision import models
from PIL import Image
from ppq.executor.torch import TorchExecutor
from ppq.IR import BaseGraph
from tqdm import tqdm
from typing import Iterable, List, Tuple
import numpy as np

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
    label_map = {}
    with open(val_file, 'r') as f:
        for line in f:
            filename, label = line.strip().split()
            label_map[os.path.splitext(filename)[0]] = int(label)

    # Collect data paths
    data_files = sorted([
        os.path.join(data_dir, file) for file in os.listdir(data_dir)
        if file.endswith('.npy') or file.endswith('.JPEG')
    ])

    # Extract base names for matching
    data_file_names = {os.path.splitext(os.path.basename(file))[0]: file for file in data_files}

    # Ensure labels match data files
    unmatched_labels = []
    unmatched_files = []

    # Filter out unmatched labels
    filtered_label_map = {}
    for name, label in label_map.items():
        if name in data_file_names:
            filtered_label_map[name] = label
        else:
            unmatched_labels.append(name)

    # Filter out unmatched files
    matched_files = []
    for name, file in data_file_names.items():
        if name in filtered_label_map:
            matched_files.append(file)
        else:
            unmatched_files.append(name)

    if unmatched_labels:
        print(f"Warning: {len(unmatched_labels)} labels have no matching files and will be ignored.")
    if unmatched_files:
        print(f"Warning: {len(unmatched_files)} files have no matching labels and will be ignored.")

    # Sort matched files and labels
    matched_files = sorted(matched_files)
    matched_labels = [filtered_label_map[os.path.splitext(os.path.basename(file))[0]] for file in matched_files]

    # Create batches
    data_batches = []
    label_batches = []

    for i in range(0, len(matched_files), batch_size):
        batch_files = matched_files[i:i + batch_size]
        batch_labels = matched_labels[i:i + batch_size]

        # Load data
        data_batch = []
        for file in batch_files:
            if file.endswith('.npy'):
                data = torch.tensor(np.load(file), dtype=torch.float32)
            elif file.endswith('.JPEG'):
                image = Image.open(file).convert('RGB')
                data = torch.tensor(np.array(image), dtype=torch.float32)
            data_batch.append(data)

        # Append to batches
        data_batches.append(torch.stack(data_batch))
        label_batches.append(torch.tensor(batch_labels, dtype=torch.long))

    # Yield batches
    for data, labels in tqdm(zip(data_batches, label_batches), desc="Preparing batches", unit="batch"):
        yield data, labels


def evaluate_quantized_model(graph, data_dir: str, val_file: str, batch_size: int, device: str = 'cuda') -> float:
    """
    Evaluate a quantized model using the provided data directory and label file.

    Args:
        graph (BaseGraph): The quantized model to be evaluated.
        data_dir (str): Path to the image or data files.
        val_file (str): Path to the validation label file.
        batch_size (int): Number of samples per batch.
        device (str): The device to run the evaluation on ('cuda' or 'cpu').

    Returns:
        float: The classification accuracy of the model.
    """
    # Configure device
    executing_device = device if torch.cuda.is_available() else 'cpu'

    # Initialize TorchExecutor with the provided graph
    executor = TorchExecutor(graph=graph, device=executing_device)

    correct = 0
    total = 0

    # Create dataloader
    dataloader = custom_dataloader(data_dir, val_file, batch_size)

    # Evaluate the model
    with torch.no_grad():
        for data, labels in tqdm(dataloader, desc="Evaluating", unit="batch"):
            data = data.to(executing_device)
            labels = labels.to(executing_device)

            # Perform inference
            outputs = executor.forward(inputs=data, output_names=None)

            # Calculate predictions
            predictions = torch.argmax(outputs[0], dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    # Compute accuracy
    accuracy = correct / total * 100
    print(f"Model Evaluation Accuracy: {accuracy:.2f}%")
    return accuracy

def evaluate_vgg16_model(data_dir: str, val_file: str, batch_size: int, device: str = 'cuda') -> float:
    """
    Evaluate the pretrained VGG16 model using the provided data directory and label file.

    Args:
        data_dir (str): Path to the image or data files.
        val_file (str): Path to the validation label file.
        batch_size (int): Number of samples per batch.
        device (str): The device to run the evaluation on ('cuda' or 'cpu').

    Returns:
        float: The classification accuracy of the model.
    """
    # Configure device
    executing_device = device if torch.cuda.is_available() else 'cpu'

    # Load pretrained VGG16 model
    model = models.vgg16(pretrained=True).to(executing_device)
    model.eval()

    correct = 0
    total = 0

    # Create dataloader
    dataloader = custom_dataloader(data_dir, val_file, batch_size)

    # Evaluate the model
    with torch.no_grad():
        for data, labels in tqdm(dataloader, desc="Evaluating VGG16", unit="batch"):
            data = data.to(executing_device)
            labels = labels.to(executing_device)

            # Perform inference
            outputs = model(data)

            # Calculate predictions
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    # Compute accuracy
    accuracy = correct / total * 100
    print(f"Pretrained VGG16 Model Evaluation Accuracy: {accuracy:.2f}%")
    return accuracy

# Example usage:
# data_dir = '/home/wangsiyuan/ppq/working/data'
# val_file = '/path/to/val.txt'
# batch_size = 64
# graph = BaseGraph(name="quantized")
# accuracy_quantized = evaluate_quantized_model(graph, data_dir, val_file, batch_size)
# accuracy_vgg16 = evaluate_vgg16_model(data_dir, val_file, batch_size)
