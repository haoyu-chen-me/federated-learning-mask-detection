import matplotlib.pyplot as plt

configs = ["E1R20", "E5R4", "E10R2"]
accuracies = [96.98, 97.25, 96.53]

plt.figure(figsize=(8, 5))
bars = plt.bar(configs, accuracies)

# 显示数值标签
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f"{yval:.2f}%", ha='center', va='bottom')

plt.title("Federated Learning Accuracy vs Local Epochs & Rounds")
plt.xlabel("Configuration")
plt.ylabel("Accuracy (%)")
plt.ylim(95.5, 97.5)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
