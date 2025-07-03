import torch
from model import MaskCNN
from dataset import load_partitioned_datasets
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
datasets = load_partitioned_datasets("face_images", num_clients=5)
test_data = torch.utils.data.ConcatDataset(datasets)
test_loader = DataLoader(test_data, batch_size=32)

def evaluate_model(model_path):
    model = MaskCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
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


configs = {
    "E1R20": "global_model_E1R20_round_20.pt",
    "E5R4": "global_model_E5R4_round_4.pt",
    "E10R2": "global_model_E10R2_round_2.pt"
}

for name, path in configs.items():
    acc = evaluate_model(path)
    print(f"{name}: Accuracy = {acc:.4f}")
