import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def train(model, trainset, epochs, batch_size, device):
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"[Train] Epoch {epoch+1}/{epochs} Loss: {running_loss/len(trainloader):.4f}")

    return model.state_dict()


def test(model, testset, batch_size, device):
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0
    test_loss = 0.0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    avg_loss = test_loss / len(testloader)
    print(f"[Eval] Accuracy: {accuracy:.4f}, Loss: {avg_loss:.4f}")

    return accuracy, avg_loss
