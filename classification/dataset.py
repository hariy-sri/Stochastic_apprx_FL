import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from typing import List, Tuple
import os

# MedMNIST imports
try:
    from medmnist import BloodMNIST, OrganSMNIST
    MEDMNIST_AVAILABLE = True
except ImportError:
    MEDMNIST_AVAILABLE = False
    print("Warning: medmnist not available. Install with: pip install medmnist")


def get_transforms(dataset_name: str, is_train: bool = True):
    """Get appropriate transforms for each dataset"""
    if dataset_name.lower() == 'mnist':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    elif dataset_name.lower() == 'cifar10':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    
    elif dataset_name.lower() == 'bloodmnist':
        if is_train:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    elif dataset_name.lower() == 'organsmnist':
        if is_train:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.Normalize(mean=[0.4953], std=[0.2292])
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4953], std=[0.2292])
            ])
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def load_dataset_splits(dataset_name: str, data_path: str = "./data"):
    """Load train, val, and test datasets"""
    os.makedirs(data_path, exist_ok=True)
    
    train_transform = get_transforms(dataset_name, is_train=True)
    test_transform = get_transforms(dataset_name, is_train=False)
    
    if dataset_name.lower() == 'mnist':
        train_dataset = datasets.MNIST(
            root=data_path, train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.MNIST(
            root=data_path, train=False, download=True, transform=test_transform
        )
        val_dataset = None
    
    elif dataset_name.lower() == 'cifar10':
        train_dataset = datasets.CIFAR10(
            root=data_path, train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.CIFAR10(
            root=data_path, train=False, download=True, transform=test_transform
        )
        val_dataset = None
    
    elif dataset_name.lower() == 'bloodmnist':
        if not MEDMNIST_AVAILABLE:
            raise ImportError("medmnist library not available. Install with: pip install medmnist")
        
        train_dataset = BloodMNIST(split="train", download=True, size=28, transform=train_transform, root=data_path)
        val_dataset = BloodMNIST(split="val", download=True, size=28, transform=test_transform, root=data_path)
        test_dataset = BloodMNIST(split="test", download=True, size=28, transform=test_transform, root=data_path)
    
    elif dataset_name.lower() == 'organsmnist':
        if not MEDMNIST_AVAILABLE:
            raise ImportError("medmnist library not available. Install with: pip install medmnist")
        
        train_dataset = OrganSMNIST(split="train", download=True, size=28, transform=train_transform, root=data_path)
        val_dataset = OrganSMNIST(split="val", download=True, size=28, transform=test_transform, root=data_path)
        test_dataset = OrganSMNIST(split="test", download=True, size=28, transform=test_transform, root=data_path)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    if val_dataset is not None:
        return train_dataset, val_dataset, test_dataset
    else:
        return train_dataset, test_dataset


def create_near_pathological_partition(dataset, num_clients: int, seed: int = 42):
    """Create near-pathological non-IID partition with balanced data assignment"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    num_classes = len(np.unique(labels))
    total_samples = len(labels)
    
    class_to_indices = {cls: [] for cls in range(num_classes)}
    for idx, label in enumerate(labels):
        label_int = int(label.item()) if hasattr(label, 'item') else int(label)
        class_to_indices[label_int].append(idx)
    
    for cls in range(num_classes):
        np.random.shuffle(class_to_indices[cls])
    
    target_samples_per_client = total_samples // num_clients
    
    if num_classes == 8:  # BloodMNIST  
        dominant_percentage = 0.65
    elif num_classes == 10:  # MNIST, CIFAR10
        dominant_percentage = 0.75
    elif num_classes == 11:  # OrganSMNIST
        dominant_percentage = 0.70
    else:
        dominant_percentage = 0.7
    
    client_indices = [[] for _ in range(num_clients)]
    class_usage = {cls: 0 for cls in range(num_classes)}
    
    specialist_clients = min(num_clients, num_classes)
    
    for client_id in range(specialist_clients):
        class_idx = client_id % num_classes
        
        dominant_samples = int(target_samples_per_client * dominant_percentage)
        class_size = len(class_to_indices[class_idx])
        available_in_class = class_size - class_usage[class_idx]
        dominant_samples = min(dominant_samples, available_in_class)
        
        start_idx = class_usage[class_idx]
        end_idx = start_idx + dominant_samples
        client_indices[client_id].extend(class_to_indices[class_idx][start_idx:end_idx])
        class_usage[class_idx] = end_idx
        
        remaining_needed = target_samples_per_client - dominant_samples
        samples_per_other_class = remaining_needed // (num_classes - 1) if num_classes > 1 else 0
        
        samples_assigned = dominant_samples
        for other_class in range(num_classes):
            if other_class == class_idx or samples_assigned >= target_samples_per_client:
                continue
                
            available_in_other = len(class_to_indices[other_class]) - class_usage[other_class]
            samples_to_take = min(samples_per_other_class, available_in_other)
            samples_to_take = min(samples_to_take, target_samples_per_client - samples_assigned)
            
            if samples_to_take > 0:
                start_idx = class_usage[other_class]
                end_idx = start_idx + samples_to_take
                client_indices[client_id].extend(class_to_indices[other_class][start_idx:end_idx])
                class_usage[other_class] = end_idx
                samples_assigned += samples_to_take
    
    for client_id in range(specialist_clients, num_clients):
        samples_assigned = 0
        samples_per_class = target_samples_per_client // num_classes
        
        for class_idx in range(num_classes):
            available_in_class = len(class_to_indices[class_idx]) - class_usage[class_idx]
            samples_to_take = min(samples_per_class, available_in_class)
            samples_to_take = min(samples_to_take, target_samples_per_client - samples_assigned)
            
            if samples_to_take > 0:
                start_idx = class_usage[class_idx]
                end_idx = start_idx + samples_to_take
                client_indices[client_id].extend(class_to_indices[class_idx][start_idx:end_idx])
                class_usage[class_idx] = end_idx
                samples_assigned += samples_to_take
        
        for class_idx in range(num_classes):
            if samples_assigned >= target_samples_per_client:
                break
                
            available_in_class = len(class_to_indices[class_idx]) - class_usage[class_idx]
            additional_needed = target_samples_per_client - samples_assigned
            samples_to_take = min(available_in_class, additional_needed)
            
            if samples_to_take > 0:
                start_idx = class_usage[class_idx]
                end_idx = start_idx + samples_to_take
                client_indices[client_id].extend(class_to_indices[class_idx][start_idx:end_idx])
                class_usage[class_idx] = end_idx
                samples_assigned += samples_to_take
    
    for client_id in range(num_clients):
        np.random.shuffle(client_indices[client_id])
    
    return client_indices


def create_iid_partition(dataset, num_clients: int):
    """Create IID partition by randomly distributing samples"""
    total_samples = len(dataset)
    indices = np.random.permutation(total_samples)
    
    samples_per_client = total_samples // num_clients
    
    client_indices = []
    for i in range(num_clients):
        start_idx = i * samples_per_client
        if i == num_clients - 1:
            end_idx = total_samples
        else:
            end_idx = (i + 1) * samples_per_client
        
        client_indices.append(indices[start_idx:end_idx].tolist())
    
    return client_indices


def create_single_rare_partition(dataset, num_clients: int, seed: int = 42):
    """Create single rare partition where only Client 0 gets rare distribution"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    num_classes = len(np.unique(labels))
    
    class_to_indices = {cls: [] for cls in range(num_classes)}
    for idx, label in enumerate(labels):
        label_int = int(label.item()) if hasattr(label, 'item') else int(label)
        class_to_indices[label_int].append(idx)
    
    for cls in range(num_classes):
        np.random.shuffle(class_to_indices[cls])
    
    class_sizes = {cls: len(class_to_indices[cls]) for cls in range(num_classes)}
    most_populated_class = max(class_sizes, key=class_sizes.get)
    max_class_size = class_sizes[most_populated_class]
    min_class_size = min(class_sizes.values())
    
    if max_class_size == min_class_size:
        most_populated_class = 0
    
    client_indices = [[] for _ in range(num_clients)]
    remaining_indices = []
    
    # Client 0 gets rare distribution
    for cls in range(num_classes):
        class_size = len(class_to_indices[cls])
        
        if cls == most_populated_class:
            samples_for_client_0 = class_size
        else:
            samples_for_client_0 = int(class_size * 0.01)
        
        samples_for_client_0 = min(samples_for_client_0, class_size)
        client_indices[0].extend(class_to_indices[cls][:samples_for_client_0])
        
        if cls != most_populated_class:
            remaining_indices.extend(class_to_indices[cls][samples_for_client_0:])
    
    # Distribute remaining samples IID among other clients
    if len(remaining_indices) > 0 and num_clients > 1:
        np.random.shuffle(remaining_indices)
        
        other_clients = num_clients - 1
        samples_per_other_client = len(remaining_indices) // other_clients
        extra_samples = len(remaining_indices) % other_clients
        
        start_idx = 0
        for client_id in range(1, num_clients):
            extra = 1 if (client_id - 1) < extra_samples else 0
            end_idx = start_idx + samples_per_other_client + extra
            
            if start_idx < len(remaining_indices):
                client_indices[client_id].extend(remaining_indices[start_idx:min(end_idx, len(remaining_indices))])
                start_idx = end_idx
    
    for client_id in range(num_clients):
        np.random.shuffle(client_indices[client_id])
    
    return client_indices


def create_federated_datasets(
    dataset_name: str,
    num_clients: int,
    data_path: str = "./data",
    partition_type: str = "near_pathological",
    alpha: float = 0.1,
    batch_size: int = 32
) -> Tuple[List[DataLoader], List[DataLoader], DataLoader]:
    """Create federated datasets with train/val split for each client and global test set"""
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    datasets_result = load_dataset_splits(dataset_name, data_path)
    
    if len(datasets_result) == 3:
        train_dataset, val_dataset, test_dataset = datasets_result
        has_builtin_val = True
    else:
        train_dataset, test_dataset = datasets_result
        val_dataset = None
        has_builtin_val = False
    
    # Create partition for training data
    if partition_type == "iid":
        client_train_indices = create_iid_partition(train_dataset, num_clients)
    elif partition_type == "near_pathological":
        client_train_indices = create_near_pathological_partition(train_dataset, num_clients)
    elif partition_type == "single_rare":
        client_train_indices = create_single_rare_partition(train_dataset, num_clients)
    else:
        raise ValueError(f"Unknown partition type: {partition_type}")
    
    # Create validation partitions
    if has_builtin_val:
        if partition_type == "iid":
            client_val_indices = create_iid_partition(val_dataset, num_clients)
        elif partition_type == "near_pathological":
            client_val_indices = create_near_pathological_partition(val_dataset, num_clients)
        elif partition_type == "single_rare":
            client_val_indices = create_single_rare_partition(val_dataset, num_clients)
    else:
        client_val_indices = None
    
    client_train_loaders = []
    client_val_loaders = []
    
    for client_idx in range(num_clients):
        train_indices = client_train_indices[client_idx]
        
        if has_builtin_val:
            val_indices = client_val_indices[client_idx]
            client_train_dataset = Subset(train_dataset, train_indices)
            client_val_dataset = Subset(val_dataset, val_indices)
        else:
            train_indices, val_indices = create_stratified_train_val_split(
                train_dataset, train_indices, test_size=0.2, random_state=42
            )
            client_train_dataset = Subset(train_dataset, train_indices)
            client_val_dataset = Subset(train_dataset, val_indices)
        
        train_loader = DataLoader(
            client_train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            drop_last=True
        )
        val_loader = DataLoader(
            client_val_dataset, 
            batch_size=batch_size, 
            shuffle=False
        )
        
        client_train_loaders.append(train_loader)
        client_val_loaders.append(val_loader)
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return client_train_loaders, client_val_loaders, test_loader


def get_dataset_info(dataset_name: str):
    """Get dataset information"""
    if dataset_name.lower() == 'mnist':
        return {
            'num_classes': 10,
            'input_channels': 1,
            'input_size': (28, 28),
            'class_names': [str(i) for i in range(10)]
        }
    elif dataset_name.lower() == 'cifar10':
        return {
            'num_classes': 10,
            'input_channels': 3,
            'input_size': (32, 32),
            'class_names': ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                          'dog', 'frog', 'horse', 'ship', 'truck']
        }
    elif dataset_name.lower() == 'bloodmnist':
        return {
            'num_classes': 8,
            'input_channels': 3,
            'input_size': (28, 28),
            'class_names': ['basophil', 'eosinophil', 'erythroblast', 'ig', 'lymphocyte', 'monocyte', 'neutrophil', 'platelet']
        }
    elif dataset_name.lower() == 'organsmnist':
        return {
            'num_classes': 11,
            'input_channels': 1,
            'input_size': (28, 28),
            'class_names': ['bladder', 'femur-left', 'femur-right', 'heart', 'kidney-left', 
                          'kidney-right', 'liver', 'lung-left', 'lung-right', 'pancreas', 'spleen']
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def create_stratified_train_val_split(dataset, client_indices, test_size=0.2, random_state=42):
    """Create stratified train/val split that maintains class distribution"""
    if len(client_indices) == 0:
        return [], []
    
    np.random.seed(random_state)
    
    client_labels = [dataset[idx][1] for idx in client_indices]
    client_labels = np.array([int(label.item()) if hasattr(label, 'item') else int(label) 
                             for label in client_labels])
    
    unique_classes = np.unique(client_labels)
    class_to_client_indices = {}
    
    for i, global_idx in enumerate(client_indices):
        label = client_labels[i]
        if label not in class_to_client_indices:
            class_to_client_indices[label] = []
        class_to_client_indices[label].append(global_idx)
    
    train_indices = []
    val_indices = []
    
    for class_label in unique_classes:
        class_indices = class_to_client_indices[class_label]
        class_size = len(class_indices)
        
        np.random.shuffle(class_indices)
        
        val_size = int(class_size * test_size)
        train_size = class_size - val_size
        
        train_indices.extend(class_indices[:train_size])
        val_indices.extend(class_indices[train_size:])
    
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    
    return train_indices, val_indices


if __name__ == "__main__":
    print("Testing available datasets and partitions...")
    
    datasets_to_test = ['mnist', 'cifar10']
    if MEDMNIST_AVAILABLE:
        datasets_to_test.extend(['bloodmnist', 'organsmnist'])
    
    partition_types = ['iid', 'near_pathological', 'single_rare']
    
    for dataset_name in datasets_to_test:
        print(f"\nTesting {dataset_name.upper()}...")
        
        try:
            info = get_dataset_info(dataset_name)
            print(f"Classes: {info['num_classes']}, Channels: {info['input_channels']}")
            
            for partition_type in partition_types:
                train_loaders, val_loaders, test_loader = create_federated_datasets(
                    dataset_name=dataset_name,
                    num_clients=5,
                    partition_type=partition_type,
                    batch_size=32
                )
                
                print(f"  {partition_type}: {len(train_loaders)} clients created")
                
        except Exception as e:
            print(f"Error with {dataset_name}: {e}")
    
    print("Testing completed.")