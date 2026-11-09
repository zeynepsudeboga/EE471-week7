import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import random_split
from torchvision.transforms import v2

# ----- Data (from the CIFAR10 tutorial) -----
transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# ADDED: hold out 5k training images to choose hyperparameters; the test set is used only at the end
train_part, val_part = random_split(trainset, [45000, 5000],
                                    generator=torch.Generator().manual_seed(42))
valloader = torch.utils.data.DataLoader(val_part, batch_size=256, shuffle=False)


# ----- The CNN (from the tutorial) -----
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


criterion = nn.CrossEntropyLoss()


# ----- Training loop and accuracy (from the tutorial, wrapped in functions) -----
def train(net, trainloader, optimizer, epochs):
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'  epoch {epoch + 1}: average loss {running_loss / len(trainloader):.3f}')


def accuracy(net, loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


# ----- ADDED: search over the tutorial's hyperparameters -----
learning_rates = [0.001, 0.01, 0.05]
batch_sizes = [4, 32, 128]
search_epochs = 2  # same as the tutorial, so every combination gets the same short training

results = []
for lr in learning_rates:
    for batch_size in batch_sizes:
        print(f"===== lr={lr}, batch_size={batch_size} =====")
        torch.manual_seed(0)  # same starting weights for every run: a fair comparison
        net = Net()
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        trainloader = torch.utils.data.DataLoader(train_part, batch_size=batch_size, shuffle=True)
        train(net, trainloader, optimizer, search_epochs)
        val_acc = accuracy(net, valloader)
        print(f"  validation accuracy: {100 * val_acc:.1f}%")
        results.append({"lr": lr, "batch_size": batch_size, "val_accuracy": round(val_acc, 4)})

results.sort(key=lambda r: r["val_accuracy"], reverse=True)
print("\n===== Results (validation accuracy, best first) =====")
for r in results:
    print(f"lr={r['lr']:<6} batch_size={r['batch_size']:<4} accuracy={100 * r['val_accuracy']:.1f}%")

# ----- ADDED: train the best combination on all 50k images, for more epochs -----
best = results[0]
epochs = 10  # the third hyperparameter: longer than the tutorial's 2 epochs
print(f"\n===== Final model: lr={best['lr']}, batch_size={best['batch_size']}, epochs={epochs} =====")
torch.manual_seed(0)
net = Net()
optimizer = optim.SGD(net.parameters(), lr=best["lr"], momentum=0.9)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=best["batch_size"], shuffle=True)
train(net, trainloader, optimizer, epochs)
torch.save(net.state_dict(), "cifar_net_tuned.pt")

# ----- Test set, used once: baseline vs tuned -----
baseline = Net()
baseline.load_state_dict(torch.load("cifar_net.pt", weights_only=True))
baseline_acc = accuracy(baseline, testloader)
tuned_acc = accuracy(net, testloader)
print(f"\nTest accuracy  baseline: {100 * baseline_acc:.1f}%   tuned: {100 * tuned_acc:.1f}%")

with open("cifar_tuning_results.json", "w") as f:
    json.dump({"search": results,
               "best": {"lr": best["lr"], "batch_size": best["batch_size"], "epochs": epochs},
               "baseline_test_accuracy": round(baseline_acc, 4),
               "tuned_test_accuracy": round(tuned_acc, 4)}, f, indent=2)
print("Saved cifar_net_tuned.pt and cifar_tuning_results.json")