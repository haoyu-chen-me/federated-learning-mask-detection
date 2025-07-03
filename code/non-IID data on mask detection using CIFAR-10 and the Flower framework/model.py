import torch
import torch.nn as nn
import torch.nn.functional as F

class CIFARCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32,
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64,
                               kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128,
                               kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # 用 dummy_input 推断展平后的大小
        dummy_input = torch.zeros(1, 3, 32, 32)
        x = self._forward_conv(dummy_input)
        self.flatten_size = x.view(-1).size(0)

        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.fc2 = nn.Linear(512, 10)

    def _forward_conv(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def fit_config(rnd: int):
        config = {"local_epochs": 5, "batch_size": 32}
        if rnd <= 20:
            config["local_epochs"] = 1
        elif rnd <= 2:
            config["local_epochs"] = 10
        return config

    # Evaluate accuracy
    def evaluate_model(model_path):
        model = MaskCNN().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
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



























