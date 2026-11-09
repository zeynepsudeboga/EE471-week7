import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from cog import BaseRunner, Input, Path
from PIL import Image
from torchvision.transforms import v2

CLASSES = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

LOG_FILE = "/tmp/predictions.log"  # read live inside the container: tail -f /tmp/predictions.log


# Same CNN as in cifar10_tutorial.py, so the saved weights fit into it
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


# Same transform as in training: tensor, 0-1, then normalized to -1..1 per color channel
transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def prepare(image_path):
    """Turn any uploaded picture into what the model saw in training: a 3x32x32 color image."""
    img = Image.open(image_path).convert("RGB").resize((32, 32))
    return transform(img).unsqueeze(0)  # add the batch dimension -> [1, 3, 32, 32]


# Test accuracies from cifar10_tutorial.py and tune_cifar.py, shown on the website
MODELS = {
    "baseline": {"file": "cifar_net.pt", "test_accuracy": 0.0},
    "tuned": {"file": "cifar_net_tuned.pt", "test_accuracy": 0.0},
}


def predict(model, x):
    """Top-3 classes of one model for an already prepared image tensor."""
    with torch.no_grad():
        probs = model(x)[0].softmax(0)  # 10 scores -> 10 probabilities
    top = probs.topk(3)
    return {
        "prediction": CLASSES[top.indices[0]],
        "confidence": round(top.values[0].item(), 4),
        "top3": {CLASSES[i]: round(p.item(), 4) for p, i in zip(top.values, top.indices)},
    }


class Runner(BaseRunner):
    def setup(self):
        """Runs once when the container starts: load both trained models."""
        self.models = {}
        for name, info in MODELS.items():
            model = Net()
            model.load_state_dict(torch.load(info["file"], map_location="cpu", weights_only=True))
            model.eval()
            self.models[name] = model
        open(LOG_FILE, "a").close()  # create the log file so tail -f works before the first request

    def run(self, image: Path = Input(description="Picture of a plane, car, bird, cat, deer, dog, frog, horse, ship or truck")) -> dict:
        """Runs for every request: classify one image with both models."""
        x = prepare(image)
        result = {}
        for name, model in self.models.items():
            result[name] = {**predict(model, x), "test_accuracy": MODELS[name]["test_accuracy"]}

        line = f"{datetime.now():%H:%M:%S} prediction: {json.dumps(result)}"
        print(line, flush=True)
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
        return result