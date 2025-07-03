import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# CSV文件路径
LOG_CSV = "accuracy_log_CIFAR.csv"


def check_csv_exists():
    if not os.path.exists(LOG_CSV):
        raise FileNotFoundError(f"❌ 未找到 {LOG_CSV}，请先运行实验")


def plot_accuracy_curves():
    df = pd.read_csv(LOG_CSV)
    plt.figure(figsize=(10, 6))
    plt.plot(df["round"], df["global_accuracy"], label="Global Accuracy", color="#1f77b4", linewidth=2)
    for i in range(5):
        plt.plot(df["round"], df[f"client{i}_acc"], label=f"Client {i} Accuracy", linestyle="--", alpha=0.7)
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("Global and Client Accuracy over Rounds")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig("global_client_accuracy.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("📊 已生成全球与客户端准确率曲线：global_client_accuracy.png")


def plot_noniid_distribution():
    labels_per_client = {
        "Client 0": [0, 1],
        "Client 1": [2, 3, 4],
        "Client 2": [5, 6, 7, 8, 9],
        "Client 3": [5, 6, 7, 8, 9],
        "Client 4": [5, 6, 7, 8, 9]
    }
    label_counts = [[1 if i in labels else 0 for i in range(10)] for labels in labels_per_client.values()]

    fig, ax = plt.subplots(figsize=(10, 6))
    clients = list(labels_per_client.keys())
    x = np.arange(len(clients))
    width = 0.08
    for i in range(10):
        counts = [label_counts[j][i] for j in range(len(clients))]
        ax.bar(x + i * width, counts, width, label=f"Label {i}")
    ax.set_xlabel("Clients")
    ax.set_ylabel("Label Proportion")
    ax.set_title("Non-IID Label Distribution Across Clients")
    ax.set_xticks(x + width * 4.5)
    ax.set_xticklabels(clients)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("non_iid_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("📊 已生成非IID标签分布图：non_iid_distribution.png")


def plot_config_comparison():
    configs = ["E1R20", "E5R4", "E10R2"]
    accuracies = [0.96925, 0.97225, 0.96875]  # 替换为真实数据
    plt.figure(figsize=(8, 5))
    bars = plt.bar(configs, accuracies, color=["#1f77b4", "#ff7f0e", "#2ca02c"], label=configs)
    plt.xlabel("Configuration (Local Epochs × Rounds)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison of E1R20, E5R4, and E10R2 Configurations")
    plt.ylim(0.95, 1.0)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, f"{yval:.4f}", ha="center", va="bottom")
    plt.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig("config_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("📊 已生成配置对比图：config_comparison.png")


if __name__ == "__main__":
    try:
        check_csv_exists()
        plot_accuracy_curves()
        plot_noniid_distribution()
        plot_config_comparison()
        print("🎉 所有可视化图表生成完成")
    except Exception as e:
        print(f"❌ 可视化失败: {e}")