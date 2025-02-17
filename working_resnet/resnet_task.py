import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50, resnet18
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    test_dataset = torchvision.datasets.ImageFolder(root=test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print(f"Test dataset size: {len(test_dataset)}")
    return test_loader

def load_pretrained_resnet18(device):
    """
    加载预训练的 ResNet-50 模型。

    Args:
        device (torch.device): 设备

    Returns:
        torch.nn.Module: 已加载的 ResNet-50 模型
    """
    model = resnet18(pretrained=True)
    model = model.to(device)
    model.eval()  # 设置为评估模式
    return model

def evaluate_model_pt(model, test_loader, device):
    """
    在测试集上评估模型性能。

    Args:
        model (torch.nn.Module): 模型
        test_loader (DataLoader): 测试数据加载器
        device (torch.device): 设备

    Returns:
        float: 准确率
    """
    model.eval()  # 确保模型在评估模式
    correct = 0
    total = 0
    with torch.no_grad():  # 禁用梯度计算
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Accuracy on the test dataset: {accuracy:.2f}%")
    return accuracy

def evaluate_model_engine(model, test_loader, device):
    """
    在测试集上评估模型性能。

    Args:
        model (torch.nn.Module): 模型
        test_loader (DataLoader): 测试数据加载器
        device (torch.device): 设备

    Returns:
        float: 准确率
    """
    model.eval()  # 确保模型在评估模式
    correct = 0
    total = 0
    with torch.no_grad():  # 禁用梯度计算
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            # 计算准确率
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Accuracy on the test dataset: {accuracy:.2f}%")
    return accuracy