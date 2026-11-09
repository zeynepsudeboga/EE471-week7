import os

from PIL import Image
from torchvision import datasets

classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Without a transform, the dataset gives PIL images we can save directly
test_data = datasets.FashionMNIST(root="data", train=False, download=True)

os.makedirs("samples", exist_ok=True)
for i in range(20):
    img, label = test_data[i]
    name = classes[label].replace("/", "-")  # "/" can't be in a file name
    # 28x28 is tiny on screen, so enlarge to 280x280 (NEAREST keeps the pixels sharp)
    img.resize((280, 280), Image.NEAREST).save(f"samples/{i:02d}_{name}.png")
print("Saved 20 sample images to samples/")