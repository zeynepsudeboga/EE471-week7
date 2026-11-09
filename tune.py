import json

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import v2

# ----- Prerequisite code (from the "Optimizing Model Parameters" tutorial) -----
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# ADDED: hold out 10k training images for choosing hyperparameters,
# so the test set is not used for the choice.
train_part, val_part = random_split(training_data, [50000, 10000],
                                    generator=torch.Generator().manual_seed(42))

# ----- Training and test loops (from the tutorial) -----
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct  # ADDED: return the accuracy so we can compare runs


loss_fn = nn.CrossEntropyLoss()

# ----- ADDED: hyperparameter search over the tutorial's hyperparameters -----
learning_rates = [1e-3, 1e-2, 1e-1]
batch_sizes = [32, 64, 128]
epochs = 5

results = []
for learning_rate in learning_rates:
    for batch_size in batch_sizes:
        print(f"===== learning_rate={learning_rate}, batch_size={batch_size} =====")
        torch.manual_seed(0)  # same starting weights for every run: a fair comparison
        model = NeuralNetwork()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        train_dataloader = DataLoader(train_part, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_part, batch_size=batch_size)
        for t in range(epochs):
            train_loop(train_dataloader, model, loss_fn, optimizer)
        accuracy = test_loop(val_dataloader, model, loss_fn)  # scored on validation data
        results.append({"learning_rate": learning_rate, "batch_size": batch_size,
                        "val_accuracy": round(accuracy, 4)})

results.sort(key=lambda r: r["val_accuracy"], reverse=True)
print("\n===== Results (validation accuracy, best first) =====")
for r in results:
    print(f"learning_rate={r['learning_rate']:<6} batch_size={r['batch_size']:<4} "
          f"accuracy={100*r['val_accuracy']:.1f}%")

# ----- ADDED: train the best combination on all 60k images, for more epochs -----
learning_rate, batch_size = results[0]["learning_rate"], results[0]["batch_size"]
epochs = 15  # the third hyperparameter: longer training than the tutorial's 5-10 epochs
print(f"\n===== Final model: learning_rate={learning_rate}, batch_size={batch_size}, epochs={epochs} =====")

torch.manual_seed(0)
model = NeuralNetwork()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_accuracy = test_loop(test_dataloader, model, loss_fn)
print("Done!")

torch.save(model.state_dict(), "model_tuned.pth")
with open("tuning_results.json", "w") as f:
    json.dump({"search": results,
               "best": {"learning_rate": learning_rate, "batch_size": batch_size, "epochs": epochs},
               "tuned_test_accuracy": round(test_accuracy, 4),
               "baseline_test_accuracy": 0.646}, f, indent=2)
print("Saved model_tuned.pth and tuning_results.json")