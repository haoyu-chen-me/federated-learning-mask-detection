import os
import flwr as fl
import torch
import warnings
from datetime import datetime

# 忽略非必要警告
warnings.filterwarnings("ignore")

# === 日志文件设置 ===
LOG_CSV = "accuracy_log_CIFAR.csv"

# 备份旧日志文件
if os.path.exists(LOG_CSV):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        os.rename(LOG_CSV, f"backup_{timestamp}_{LOG_CSV}")
        print(f"📁 旧日志已备份为 backup_{timestamp}_{LOG_CSV}")
    except PermissionError:
        print("⚠️ 日志文件被占用，未能备份旧内容")

# 写入表头
with open(LOG_CSV, "w") as f:
    f.write("round,global_accuracy,client0_acc,client1_acc,client2_acc,client3_acc,client4_acc\n")

# === 自定义策略：记录全局和客户端准确率 ===
class SaveMetricsStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        if failures:
            print(f"❌ Round {server_round} failed.")
            return None

        total_examples = 0
        total_correct = 0
        client_accuracies = []

        for client_proxy, fit_res in results:
            print(f"[Server] Received metrics: {fit_res.metrics}")
            total_examples += fit_res.num_examples
            total_correct += fit_res.metrics.get("correct", 0)
            # 计算每个客户端的准确率
            client_acc = fit_res.metrics.get("correct", 0) / fit_res.num_examples
            client_accuracies.append(client_acc)

        if total_examples > 0:
            global_accuracy = total_correct / total_examples
            print(f"✅ [Round {server_round}] Global Accuracy: {global_accuracy:.4f}")
            # 保存全局和客户端准确率
            with open(LOG_CSV, "a") as f:
                client_acc_str = ",".join([f"{acc:.4f}" for acc in client_accuracies])
                f.write(f"{server_round},{global_accuracy:.4f},{client_acc_str}\n")

        return super().aggregate_fit(server_round, results, failures)

# === 每轮传参：本地 epoch 数等 ===
def fit_config(server_round):
    return {
        "local_epochs": 5,
        "batch_size": 32,
    }

# === 启动联邦服务器 ===
if __name__ == "__main__":
    print("🚀 启动联邦服务器...")
    try:
        fl.server.start_server(
            server_address="localhost:8080",
            config=fl.server.ServerConfig(num_rounds=10),
            strategy=SaveMetricsStrategy(
                fraction_fit=1.0,
                min_fit_clients=5,
                min_available_clients=5,
                on_fit_config_fn=fit_config,
            )
        )
        print("🎉 服务器运行完成。")
    except Exception as e:
        print(f"❌ 服务器运行失败: {e}")