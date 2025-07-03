import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset
import numpy as np
import random


def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


def split_cifar10_noniid_label_skew(num_clients=5, seed=42):
    transform = get_transform()
    dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    labels = np.array(dataset.targets)

    # 标签分布规则：
    # 20% 客户端：标签 0-1
    # 30% 客户端：标签 2-4
    # 50% 客户端：标签 5-9
    class_partitions = {
        0: [0, 1],                  # 1个客户端
        1: [2, 3, 4],               # 1个客户端
        2: [5, 6, 7, 8, 9],         # 3个客户端
    }

    # 分配比例
    client_class_map = []
    client_count = 0
    for group_id, class_list in class_partitions.items():
        group_clients = int([0.2, 0.3, 0.5][group_id] * num_clients)
        for _ in range(group_clients):
            client_class_map.append(class_list)
            client_count += 1

    # 补足不足部分
    while len(client_class_map) < num_clients:
        client_class_map.append([5, 6, 7, 8, 9])

    random.seed(seed)
    np.random.seed(seed)

    client_datasets = []
    for class_subset in client_class_map:
        idx = [i for i, label in enumerate(labels) if label in class_subset]
        sampled_idx = np.random.choice(idx, size=min(len(idx), 1000), replace=False)
        client_datasets.append(Subset(dataset, sampled_idx))

    return client_datasets
