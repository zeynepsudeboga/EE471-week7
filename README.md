# EE471 – Week 7: Image classification with PyTorch, Cog and Docker

Two image classifiers, each trained twice (tutorial settings vs. tuned hyperparameters), containerized with Cog and served by a website that compares both models side by side.

| | Dataset | Model | Baseline test acc. | Tuned test acc. |
|---|---|---|---|---|
| Task 1 | FashionMNIST (28×28 grayscale garments) | Fully connected network from the PyTorch Quickstart | 64.6% | **89.4%** |
| Task 2 | CIFAR-10 (32×32 color photos) | CNN from the PyTorch CIFAR-10 tutorial | 53.7% | **60.1%** |

## Repository layout

| Path | What it is |
|---|---|
| `fashion_quickstart.py` | PyTorch Quickstart: trains the FashionMNIST baseline → `model.pth` |
| `tune.py` | Hyperparameter search for FashionMNIST → `model_tuned.pth`, `tuning_results.json` |
| `cifar10_tutorial.py` | PyTorch CIFAR-10 tutorial: trains the CNN baseline → `cifar_net.pt` |
| `tune_cifar.py` | Hyperparameter search for CIFAR-10 → `cifar_net_tuned.pt`, `cifar_tuning_results.json` |
| `export_samples.py`, `export_cifar_samples.py` | Save 20 test-set images as PNG (`samples/`, `cifar_samples/`) for demos |
| `fashion-classifier/`, `cifar-classifier/` | Cog model (`cog.yaml`, `run.py`, weights) + website (`web/`) + `compose.yaml` |

## Setup

Requirements: Python 3, [Docker Desktop](https://www.docker.com/products/docker-desktop/), [Cog](https://github.com/replicate/cog) (`brew install replicate/tap/cog`).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision matplotlib
```

## Train (optional – the trained weights are already in the repo)

```bash
python fashion_quickstart.py   # baseline, 5 epochs
python tune.py                 # grid search + tuned model (~10–15 min on CPU)
python cifar10_tutorial.py     # baseline CNN, 2 epochs (close the image window to start training)
python tune_cifar.py           # grid search + tuned CNN (~10–20 min on CPU)
```
The datasets are downloaded automatically into `data/` (CIFAR-10 is ~170 MB and its server can be slow).

Hyperparameters are chosen on a validation split of the training data; the test set is used only to report the final accuracy.

## Run a classifier website

```bash
cd fashion-classifier            # or: cd cifar-classifier
cog build -t fashion-classifier  # or: cog build -t cifar-classifier
docker compose up --build
```
- Website: http://localhost:8000 – upload an image, press **Classify**, compare the baseline and tuned models.
- Model API: `POST http://localhost:5050/predictions` with `{"input": {"image": "data:image/png;base64,..."}}`
- Live prediction logs from inside the container:
  ```bash
  docker exec -it fashion-model bash   # or: cifar-model
  tail -f /tmp/predictions.log
  ```
Only one classifier can run at a time, because both use ports 8000 and 5050.

## Notes
- FashionMNIST images are light garments on a black background, so `run.py` converts uploads to 28×28 grayscale and inverts photos with a light background. Real photos still differ a lot from the training data; the images in `samples/` work best.
- CIFAR-10 models work on real photos, but objects should fill the picture and have a natural background.
