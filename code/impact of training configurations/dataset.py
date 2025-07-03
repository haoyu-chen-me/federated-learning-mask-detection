import os
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import random

class FaceMaskDataset(Dataset):
    def __init__(self, root_dir, samples):
        self.root_dir = root_dir
        self.samples = samples
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 确保图像尺寸统一
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sub_path, label = self.samples[idx]
        img_path = os.path.join(self.root_dir, sub_path)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label

def load_partitioned_datasets(root_dir, num_clients=5):
    samples = []

    # 读取 masked 图像
    masked_dir = os.path.join(root_dir, "masked")
    for fname in os.listdir(masked_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            samples.append((f"masked/{fname}", 1))

    # 读取 unmasked 图像
    unmasked_dir = os.path.join(root_dir, "unmasked")
    for fname in os.listdir(unmasked_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            samples.append((f"unmasked/{fname}", 0))

    # 洗牌
    random.seed(42)
    random.shuffle(samples)

    # 平均划分给每个客户端
    total = len(samples)
    per_client = total // num_clients

    client_datasets = []
    for i in range(num_clients):
        start = i * per_client
        end = total if i == num_clients - 1 else (i + 1) * per_client
        subset = samples[start:end]
        client_datasets.append(FaceMaskDataset(root_dir, subset))

    return client_datasets
