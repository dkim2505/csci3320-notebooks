"""Build lab13_1_mnist_mlp.ipynb, CSCI 3320 Lab 13-1 (Topic 13, Thu Oct 15 2026).

The student changes ONE cell, the model. Everything else is given: the data (MNIST
from torchvision), the training loop that records a learning curve, and the plot.
The starter model is 784-10 (no hidden layer), which runs as given and scores about
0.92; the lab is to add hidden layers and compare the curves.

    python3 build_lab13_1.py
"""
import json

META = {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(True)}


DATA = '''
# ---------------- GIVEN: the data. Run it, do not change it. ----------------
import torch, torch.nn as nn
import torchvision
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print("running on", device)

# MNIST: 60,000 training and 10,000 test images of handwritten digits, 28 x 28 pixels.
train = torchvision.datasets.MNIST("data", train=True, download=True)
test = torchvision.datasets.MNIST("data", train=False, download=True)

# Flatten each image to 784 numbers in [0, 1]: a network takes a list of numbers.
X_train = train.data.reshape(-1, 784).float().div(255).to(device)   # (60000, 784)
y_train = train.targets.to(device)                                   # (60000,) digits 0 to 9
X_test = test.data.reshape(-1, 784).float().div(255).to(device)     # (10000, 784)
y_test = test.targets.to(device)
print("train", tuple(X_train.shape), " test", tuple(X_test.shape))
'''

MODEL = '''
# Lab 13-1
# Name: Your Full Name
# ======================= YOUR PART: the network =======================
# Rules of the sizes (Topic 13):
#   - the first layer takes 784 numbers in (one per pixel),
#   - the last layer gives 10 numbers out (one score per digit),
#   - each nn.Linear(a, b)'s output size b is the next nn.Linear's input size,
#   - put nn.ReLU() between two nn.Linear layers (without it, two layers act like one).
#
# The starter is 784-10: no hidden layer at all. It runs as it is.
# Change it into a network WITH hidden layers, for example 784-128-10:
#     nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10)
# or 784-128-64-10, or a design of your own.

model = nn.Sequential(
    nn.Linear(784, 10),
)

# =======================================================================
model = model.to(device)
print(model)
print("parameters:", sum(p.numel() for p in model.parameters()))
'''

TRAIN = '''
# ---------------- GIVEN: training, with a learning curve. Do not change it. ----------------
EPOCHS = 10           # passes over the 60,000 training images
BATCH = 128           # images per gradient step (a mini-batch)

torch.manual_seed(0)
for layer in model:   # start every run from fresh random weights
    if hasattr(layer, "reset_parameters"):
        layer.reset_parameters()
loss_fn = nn.CrossEntropyLoss()                          # softmax + cross-entropy
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

def accuracy(X, y):
    model.eval()
    with torch.no_grad():
        return (model(X).argmax(dim=1) == y).float().mean().item()

history = {"loss": [], "train": [], "test": []}
for epoch in range(EPOCHS):
    model.train()
    order = torch.randperm(len(X_train), device=device)  # a new shuffle every epoch
    total = 0.0
    for i in range(0, len(X_train), BATCH):
        idx = order[i:i + BATCH]
        loss = loss_fn(model(X_train[idx]), y_train[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * len(idx)
    history["loss"].append(total / len(X_train))
    history["train"].append(accuracy(X_train, y_train))
    history["test"].append(accuracy(X_test, y_test))
    print(f"epoch {epoch + 1:2d}  loss {history['loss'][-1]:.4f}  "
          f"train acc {history['train'][-1]:.4f}  test acc {history['test'][-1]:.4f}")
'''

PLOT = '''
# ---------------- GIVEN: the learning curves. Do not change it. ----------------
sizes = [m.in_features for m in model if isinstance(m, nn.Linear)] + [10]
shape = "-".join(str(s) for s in sizes)
epochs = range(1, EPOCHS + 1)

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].plot(epochs, history["loss"], "o-")
ax[0].set_title("training loss"); ax[0].set_xlabel("epoch")
ax[1].plot(epochs, history["train"], "o-", label="train")
ax[1].plot(epochs, history["test"], "o-", label="test")
ax[1].set_title("accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
fig.suptitle(f"MNIST, network {shape}, {sum(p.numel() for p in model.parameters()):,} parameters")
plt.tight_layout(); plt.show()
print(f"network {shape}: final test accuracy {history['test'][-1]:.4f}")
'''

CELLS = [
    md("""
# Lab 13-1: A neural network on MNIST, with a learning curve

CSCI 3320 · Topic 13 · Neural Networks I

You will train a neural network on **MNIST**, 70,000 handwritten digits, and watch its
**learning curve**: the training loss, and the training and test accuracy, after every epoch.

**You change one cell: the network.** Everything else is given and must stay as it is:
the data, the training loop, and the plot.

The starter network is **784-10**: 784 inputs (one per pixel) straight to 10 outputs
(one per digit), with **no hidden layer**. That is softmax regression from Topic 11b. It runs
as given and reaches about **0.92** on the test digits. Your job is to give it **hidden layers**
and see what they do to the curve.

**Before you start:** Runtime > Change runtime type > **T4 GPU** (it also runs on a CPU, just slower).
"""),
    md("## Step 1. The data (given)\n\nRun this cell. It downloads MNIST the first time (about 10 MB)."),
    code(DATA),
    md("""
## Step 2. The network (your part)

1. Put **your full name** in the second comment line.
2. Run the cell **as it is** first, then run Steps 3 and 4: that is the 784-10 baseline.
   Note its test accuracy.
3. Now change the network to one **with at least one hidden layer**, for example 784-128-10,
   and run Steps 2, 3 and 4 again.
"""),
    code(MODEL),
    md("## Step 3. Training (given)\n\nTen epochs. On a T4 GPU this takes well under a minute; on a CPU, a few minutes."),
    code(TRAIN),
    md("## Step 4. The learning curve (given)"),
    code(PLOT),
    md("""
## What to submit

One screenshot of **your network with hidden layers**, showing together:

- the **Step 2 cell** with your full name and your network,
- the **learning curve** from Step 4, with the final test accuracy line under it.

Upload it to the **Lab 13-1** dropbox on Brightspace. Graded on completion.

## Think about it (not graded)

- How much did the hidden layer add over 784-10?
- Compare the train and test curves. Where do they start to separate, and what does that mean (Topic 10)?
- Try a second design (wider, deeper). Does a bigger network always score higher on the test digits?
"""),
]

if __name__ == "__main__":
    with open("lab13_1_mnist_mlp.ipynb", "w") as f:
        json.dump({"cells": CELLS, "metadata": META, "nbformat": 4, "nbformat_minor": 0}, f, indent=1)
    print("wrote lab13_1_mnist_mlp.ipynb", len(CELLS), "cells")
