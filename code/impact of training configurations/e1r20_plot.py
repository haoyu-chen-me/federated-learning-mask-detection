# server_e1r20.py
import flwr as fl
from model import MaskCNN
import torch
import csv
from dataset import load_partitioned_datasets
from torch.utils.data import DataLoader
from flwr.common import parameters_to_ndarrays

# 构造测试集
datasets = load_partitioned_datasets("face_images", num_clients=5)
test_data = torch.utils.data.ConcatDataset(datasets)
test_loader = DataLoader(test_data, batch_size=32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义准确率评估函数
def evaluate_model(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    return correct / total

# 自定义策略
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.final_parameters = None
        self.model = MaskCNN().to(device)
        self.accuracy_log = []

    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters, _ = super().aggregate_fit(rnd, results, failures)

        if aggregated_parameters is not None:
            weights = parameters_to_ndarrays(aggregated_parameters)
            params_dict = zip(self.model.state_dict().keys(), weights)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            self.model.load_state_dict(state_dict, strict=True)

            # 保存模型
            torch.save(self.model.state_dict(), f"global_model_round_{rnd}.pt")
            print(f"✅ Saved global model to global_model_round_{rnd}.pt")

            # ✅ 每轮后评估一次准确率
            acc = evaluate_model(self.model)
            self.accuracy_log.append((rnd, acc))
            print(f"📈 Round {rnd} Accuracy: {acc:.4f}")

            # ✅ 每轮写入 CSV
            with open("accuracy_log_E1R20.csv", mode="a", newline="") as f:
                writer = csv.writer(f)
                if rnd == 1:
                    writer.writerow(["Round", "Accuracy"])
                writer.writerow([rnd, acc])

        return aggregated_parameters, {}

# 配置本地 epoch
def fit_config(rnd: int):
    return {"local_epochs": 1}

strategy = SaveModelStrategy(
    fraction_fit=1.0,
    min_fit_clients=5,
    min_available_clients=5,
    on_fit_config_fn=fit_config,
)

if __name__ == "__main__":
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=20),
        strategy=strategy
    )
