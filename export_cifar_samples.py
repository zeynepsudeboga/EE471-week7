import os

from PIL import Image
import torchvision

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Without a transform, the dataset gives PIL images we can save directly
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

os.makedirs("cifar_samples", exist_ok=True)
for i in range(20):
    img, label = testset[i]
    # 32x32 is tiny on screen, so enlarge to 256x256 (NEAREST keeps the pixels sharp)
    img.resize((256, 256), Image.NEAREST).save(f"cifar_samples/{i:02d}_{classes[label]}.png")
print("Saved 20 sample images to cifar_samples/")