import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import copy


class MNISTNetReLU(nn.Module):
    """LeNet-5 with ReLU activations for MNIST (28x28, 1 channel, 10 classes)"""
    def __init__(self, num_classes: int = 10):
        super(MNISTNetReLU, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=4, stride=1, padding=0)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.maxpool1(F.relu(self.conv1(x)))
        x = self.maxpool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ResNet9_GN_Adaptive(nn.Module):
    """ResNet-9 with Group Normalization and Adaptive Pooling"""
    def __init__(self, num_classes: int, in_channels: int, dim: int, num_groups: int = 32) -> None:
        super(ResNet9_GN_Adaptive, self).__init__()
        self.num_groups = num_groups

        self.prep = self._make_layer(in_channels, 64)
        self.layer1_head = self._make_layer(64, 128, pool=True)
        self.layer1_residual = nn.Sequential(
            self._make_layer(128, 128), 
            self._make_layer(128, 128)
        )
        self.layer2 = self._make_layer(128, 256, pool=True)
        self.layer3_head = self._make_layer(256, 512, pool=True)
        self.layer3_residual = nn.Sequential(
            self._make_layer(512, 512), 
            self._make_layer(512, 512)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(dim, num_classes)

    def _make_layer(self, in_channels, out_channels, pool=False):
        layers = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(self.num_groups, out_channels),
            nn.ReLU(inplace=True)
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1_head(x)
        x = self.layer1_residual(x) + x
        
        x = self.layer2(x)
        x = self.layer3_head(x)
        x = self.layer3_residual(x) + x
        
        x1 = self.adaptive_pool(x)
        x2 = x1.view(x1.size(0), -1)
        x3 = self.linear(x2)
        return x3


class CIFAR10ResNet9(nn.Module):
    """ResNet-9 for CIFAR-10 dataset (32x32, 3 channels, 10 classes)"""
    def __init__(self, num_classes: int = 10, num_groups: int = 32):
        super(CIFAR10ResNet9, self).__init__()
        self.resnet9 = ResNet9_GN_Adaptive(num_classes=num_classes, in_channels=3, dim=512, num_groups=num_groups)
        
    def forward(self, x):
        return self.resnet9(x)


class BloodMNISTResNet9(nn.Module):
    """ResNet-9 for BloodMNIST dataset (28x28, 3 channels, 8 classes)"""
    def __init__(self, num_classes: int = 8, num_groups: int = 32):
        super(BloodMNISTResNet9, self).__init__()
        self.resnet9 = ResNet9_GN_Adaptive(num_classes=num_classes, in_channels=3, dim=512, num_groups=num_groups)
        
    def forward(self, x):
        return self.resnet9(x)


class OrganSMNISTResNet9(nn.Module):
    """ResNet-9 for OrganSMNIST dataset (28x28, 1 channel, 11 classes)"""
    def __init__(self, num_classes: int = 11, num_groups: int = 32):
        super(OrganSMNISTResNet9, self).__init__()
        self.resnet9 = ResNet9_GN_Adaptive(num_classes=num_classes, in_channels=1, dim=512, num_groups=num_groups)
        
    def forward(self, x):
        return self.resnet9(x)


def get_model(dataset_name: str, num_classes: int = None):
    """Factory function to get appropriate model based on dataset"""
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'mnist':
        return MNISTNetReLU(num_classes=num_classes or 10)
    elif dataset_name == 'cifar10':
        return CIFAR10ResNet9(num_classes=num_classes or 10)
    elif dataset_name == 'bloodmnist':
        return BloodMNISTResNet9(num_classes=num_classes or 8)
    elif dataset_name == 'organsmnist':
        return OrganSMNISTResNet9(num_classes=num_classes or 11)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_model_parameters(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Get model parameters as dictionary"""
    return {name: param.data.clone() for name, param in model.named_parameters()}


def set_model_parameters(model: nn.Module, parameters: Dict[str, torch.Tensor]):
    """Set model parameters from dictionary"""
    for name, param in model.named_parameters():
        param.data = parameters[name].clone()


def calculate_model_difference(model1: nn.Module, model2: nn.Module) -> Dict[str, torch.Tensor]:
    """Calculate difference between two models"""
    diff = {}
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        assert name1 == name2, f"Parameter names don't match: {name1} vs {name2}"
        diff[name1] = param1.data - param2.data
    return diff


def apply_model_difference(model: nn.Module, diff: Dict[str, torch.Tensor], alpha: float = 1.0):
    """Apply model difference with scaling factor"""
    for name, param in model.named_parameters():
        param.data = param.data + alpha * diff[name]


def model_parameters_to_vector(model: nn.Module) -> torch.Tensor:
    """Convert model parameters to a single vector"""
    return torch.cat([param.data.flatten() for param in model.parameters()])


def vector_to_model_parameters(model: nn.Module, vector: torch.Tensor):
    """Set model parameters from a vector"""
    idx = 0
    for param in model.parameters():
        param_length = param.numel()
        param.data = vector[idx:idx + param_length].reshape(param.shape)
        idx += param_length


if __name__ == "__main__":
    print("Testing model creation...")
    
    datasets = ['mnist', 'cifar10', 'bloodmnist', 'organsmnist']
    
    for dataset_name in datasets:
        try:
            model = get_model(dataset_name)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"{dataset_name}: {total_params:,} parameters")
        except Exception as e:
            print(f"Error with {dataset_name}: {e}")