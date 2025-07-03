import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import MaskCNN
from dataset import load_partitioned_datasets
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载客户端数据
CLIENT_ID = int(os.environ.get("CLIENT_ID", 0))
datasets = load_partitioned_datasets("face_images", num_clients=5)
trainset = datasets[CLIENT_ID]
trainloader = DataLoader(trainset, batch_size=16, shuffle=True)

# 定义客户端类
class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = MaskCNN().to(DEVICE)

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for k, v in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(v)
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        LOCAL_EPOCH = int(config.get("local_epochs", 1))  # 从 server 传入
        for epoch in range(LOCAL_EPOCH):
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        return self.get_parameters(), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else 0.0
        return avg_loss, total, {"accuracy": accuracy}

if __name__ == "__main__":
    fl.client.start_client(
        server_address="localhost:8080",
        client=FlowerClient().to_client()
    )

