import json
from datetime import datetime

import torch
from cog import BaseRunner, Input, Path
from PIL import Image
from torch import nn
from torchvision.transforms import v2

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

LOG_FILE = "/tmp/predictions.log"  # read live inside the container: tail -f /tmp/predictions.log


# Same network as in fashion_quickstart.py, so the saved weights fit into it
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# Same transform as in training: image -> tensor -> float values between 0 and 1
transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])


def prepare(image_path):
    """Turn any uploaded picture into what the model saw in training: 1x28x28, light garment on black."""
    img = Image.open(image_path).convert("L").resize((28, 28))  # grayscale, 28x28
    x = transform(img)                                          # shape [1, 28, 28]
    # FashionMNIST garments are light on a black background. Photos are usually the
    # opposite (dark object on a light background), so invert those.
    border = torch.cat([x[0, 0, :], x[0, -1, :], x[0, :, 0], x[0, :, -1]])
    if border.mean() > 0.5:
        x = 1 - x
    return x.unsqueeze(0)  # add the batch dimension -> [1, 1, 28, 28]


# Test accuracies measured in fashion_quickstart.py and tune.py, shown on the website
MODELS = {
    "baseline": {"file": "model.pth", "test_accuracy": 0.646},
    "tuned": {"file": "model_tuned.pth", "test_accuracy": 0.894},
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
            model = NeuralNetwork()
            model.load_state_dict(torch.load(info["file"], map_location="cpu", weights_only=True))
            model.eval()
            self.models[name] = model
        open(LOG_FILE, "a").close()  # create the log file so tail -f works before the first request

    def run(self, image: Path = Input(description="Picture of a garment")) -> dict:
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