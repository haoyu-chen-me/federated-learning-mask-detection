import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train_utils import train
from model import CIFARCNN
from dataset import split_cifar10_noniid_label_skew
import flwr as fl
import sys

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 客户端编号（用于分配数据）
CLIENT_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0

# 加载非IID数据（标签倾斜）
client_datasets = split_cifar10_noniid_label_skew(num_clients=5)
trainset = client_datasets[CLIENT_ID]
testset = client_datasets[CLIENT_ID]  # 默认测试数据与训练相同（你可自定义）

# Flower 客户端定义
class CifarClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = CIFARCNN().to(DEVICE)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        new_dict = dict(zip(state_dict.keys(), parameters))
        self.model.load_state_dict({k: torch.tensor(v) for k, v in new_dict.items()})

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        local_epochs = config.get("local_epochs", 1)
        print(f"[Client {CLIENT_ID}] Training for {local_epochs} local epochs...")

        updated_weights = train(self.model, trainset, local_epochs, batch_size=32, device=DEVICE)

        self.model.eval()
        testloader = DataLoader(testset, batch_size=32, shuffle=False)
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(images)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        print(f"[Client {CLIENT_ID}] Local Accuracy after fit: {acc:.4f} ({correct}/{total})")

        return (
            [val.cpu().numpy() for val in updated_weights.values()],
            len(trainset),
            {"correct": correct},
        )

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.to(DEVICE)
        self.model.eval()

        testloader = DataLoader(testset, batch_size=32, shuffle=False)
        criterion = nn.CrossEntropyLoss()
        correct = 0
        total = 0
        loss_total = 0.0

        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss_total += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_loss = loss_total / len(testloader)
        accuracy = correct / total
        print(f"[Client {CLIENT_ID} Eval] Accuracy: {accuracy:.4f} ({correct}/{total})")

        return avg_loss, total, {"correct": correct}

if __name__ == "__main__":
    print(f"🚀 启动客户端 {CLIENT_ID}")
    try:
        fl.client.start_numpy_client(server_address="localhost:8080", client=CifarClient())
        print(f"🎉 客户端 {CLIENT_ID} 运行完成。")
    except Exception as e:
        print(f"❌ 客户端 {CLIENT_ID} 运行失败: {e}")