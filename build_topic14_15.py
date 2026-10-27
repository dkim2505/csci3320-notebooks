"""Build the companion notebooks for CSCI 3320 Topics 14 and 15.

    topic14_pytorch_essentials.ipynb    Topic 14 (note key 13, Thu Oct 22 2026)
    topic15_designing_networks.ipynb    Topic 15 (note key 13c, Tue Oct 27 2026)

Not labs: nothing is submitted. Each notebook is the lecture note's code, cell by
cell and line for line, so students can run what the page shows and then change
it in the "Try it" cells. Both run on a CPU in seconds (Topic 15's size grid
takes about half a minute), so no GPU is asked for.

    python3 build_topic14_15.py
"""
import json

META = {"colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(True)}


def write(name, cells):
    with open(name, "w") as f:
        json.dump({"cells": cells, "metadata": META, "nbformat": 4, "nbformat_minor": 0}, f, indent=1)
    print("wrote", name, len(cells), "cells")


NOTE14 = "https://dongchul.kim/csci3320-fall2026/topic/13"
NOTE15 = "https://dongchul.kim/csci3320-fall2026/topic/13c"

# ------------------------------------------------------------------ Topic 14
T14 = [
    md(f"""
# Topic 14: PyTorch essentials

CSCI 3320 · Neural Networks II · tensors, autograd, the training loop

This notebook is the code from the [Topic 14 lecture note]({NOTE14}), in the same order.
Run the cells from top to bottom (Runtime > Run all works too), compare what you see
with the note, then change things in the **Try it** cells.

It runs on a CPU in a few seconds; no GPU is needed. The printed numbers can differ a
little from the note on a different PyTorch version.
"""),
    md("## 1. Tensors: NumPy arrays that remember"),
    code('''
import numpy as np
import torch

t = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])
print(t.shape, t.dtype)
print(t * 10)
print(t.sum(), t[1, 2])

a = np.array([1.0, 2.0])
print(torch.tensor(a), torch.tensor(a).numpy())
'''),
    md("""
A NumPy array of decimals becomes a `float64` tensor, but PyTorch's layers expect `float32`.
That is why the program in section 3 writes `dtype=torch.float32`.

## 2. Autograd: PyTorch finds the slope for you

Topic 8a's hand example: `x = [1, 2]`, `y = [2, 4]`, the model `w * x`, starting at `w = 0`.
By hand you found the cost 5 and the slope -5.
"""),
    code('''
import torch

x = torch.tensor([1.0, 2.0])
y = torch.tensor([2.0, 4.0])
w = torch.tensor(0.0, requires_grad=True)   # a knob: PyTorch, please track it

cost = ((w * x - y) ** 2).mean() / 2        # Topic 8a's cost, at w = 0
print(cost)
cost.backward()                             # work out the slope for every tracked knob
print(w.grad)
'''),
    md("""
### Gradients add up

Run the cost and `.backward()` again without changing `w`. The slope is still -5, but
`w.grad` now says -10: PyTorch **adds** each new gradient to `.grad`. That is why every
training loop starts with `optimizer.zero_grad()`.
"""),
    code('''
cost = ((w * x - y) ** 2).mean() / 2        # same cost again, w has not moved
cost.backward()
print(w.grad)                               # added to the old one, not replaced
'''),
    md("""
**Try it.** Run the cell above a few more times. What does `w.grad` say each time?
Then run `w.grad.zero_()` in a new cell and run the cell above once more.

## 3. The whole program: iris 4-8-3

The five steps per batch: **zero_grad, forward, loss, backward, step**.
"""),
    code('''
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

torch.manual_seed(0)
X, y = load_iris(return_X_y=True)
X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
loader = DataLoader(TensorDataset(X_train, y_train), batch_size=20, shuffle=True)

model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.02)

for epoch in range(100):
    for xb, yb in loader:
        optimizer.zero_grad()          # 1. clear old gradients
        pred = model(xb)               # 2. forward: make predictions
        loss = loss_fn(pred, yb)       # 3. measure how wrong
        loss.backward()                # 4. backward: find every gradient
        optimizer.step()               # 5. step: nudge every weight
    if (epoch + 1) % 20 == 0:
        print(f"epoch {epoch + 1:3d}   loss {loss.item():.4f}")

model.eval()
with torch.no_grad():
    acc = (model(X_test).argmax(dim=1) == y_test).float().mean()
print(f"test accuracy: {acc.item():.4f}")
'''),
    md("""
The note's run prints a loss of about 0.85 after 20 epochs, 0.22 after 100, and a test
accuracy of 1.0000 (30 of 30 flowers).

**Try it.** Change `lr=0.02` to `0.002` and to `0.5`, run the cell again each time, and
watch the loss. Then put it back.

## 4. When a step is missing or out of order

The same program, with one thing broken at a time. Every version uses the same seed and
the same data, so only the mistake differs. PyTorch raises **no error** for any of them.
"""),
    code('''
def run(variant):
    torch.manual_seed(0)
    X, y = load_iris(return_X_y=True)
    X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=20, shuffle=True)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
    losses = []
    for epoch in range(100):
        for xb, yb in loader:
            if variant != "no zero_grad":
                optimizer.zero_grad()
            if variant == "step before backward":
                optimizer.step()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            if variant != "no backward":
                loss.backward()
            if variant not in ("step before backward", "no step"):
                optimizer.step()
        if (epoch + 1) % 20 == 0:
            losses.append(round(loss.item(), 2))
    model.eval()
    with torch.no_grad():
        acc = (model(X_test).argmax(dim=1) == y_test).float().mean().item()
    print(f"{variant:22s} loss {losses}  test accuracy {acc:.4f}")


for v in ["correct", "no zero_grad", "step before backward", "no backward", "no step"]:
    run(v)
'''),
    md("""
Three of the four mistakes print **identical** numbers: in all three the weights never
change, the loss stays near 1.2, and the network guesses at chance (1 in 3).
A missing `zero_grad()` makes the loss jump around instead.

## 5. Testing: `model.eval()` and `torch.no_grad()`

Outside `no_grad()` an output carries a record (`grad_fn`); inside it, there is none.
"""),
    code('''
torch.manual_seed(0)
m = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
x = torch.rand(2, 4)
print("without no_grad:", m(x).grad_fn)
with torch.no_grad():
    print("with no_grad:   ", m(x).grad_fn)
'''),
    md(f"""
## What to take away

Every PyTorch training loop is the same five lines in the same order:
**zero_grad, forward, loss, backward, step**. A flat loss means the weights are not
moving; a wild loss often means gradients are piling up. Back to the
[lecture note]({NOTE14}).
"""),
]

# ------------------------------------------------------------------ Topic 15
T15 = [
    md(f"""
# Topic 15: Designing a network in PyTorch

CSCI 3320 · Neural Networks III

This notebook is the code from the [Topic 15 lecture note]({NOTE15}), in the same order:
a network picture turned into `nn.Sequential`, the iris 4-16-3 network with its learning
curves, and three network sizes on three datasets. Run the cells from top to bottom, then
change things in the **Try it** cells.

It runs on a CPU; the longest cell (27 small training runs) takes about half a minute.
The printed numbers can differ a little from the note on a different PyTorch version.
"""),
    md("""
## 1. From a picture to code: the 4-16-3 network

- Each set of arrows is one `nn.Linear(in_features, out_features)`.
- Each hidden layer is followed by `nn.ReLU()`; the last layer has none.
- **Rule 1, neighbors match**: one layer's `out_features` is the next one's `in_features`.
- **Rule 2, the ends come from the data**: features in, classes out.
"""),
    code('''
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(4, 16),   # 4 features in, 16 hidden neurons out
    nn.ReLU(),          # bend: after every hidden layer
    nn.Linear(16, 3),   # 16 in (Rule 1), 3 classes out (Rule 2)
)
print(model)
'''),
    md("### Counting the weights\n\n`nn.Linear(a, b)` holds $a \\times b + b$ numbers."),
    code('''
for name, p in model.named_parameters():
    print(name, tuple(p.shape), p.numel())
print("total:", sum(p.numel() for p in model.parameters()))
'''),
    md("### Checking the shapes before training"),
    code('''
x = torch.zeros(5, 4)       # 5 rows, 4 features
print(model(x).shape)
'''),
    md("""
### When a rule is broken

The first layer gives 8 numbers but the next one expects 16. The `try` lets the notebook
keep going so you can read the message.
"""),
    code('''
bad = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(16, 3))   # 8, then 16
try:
    bad(torch.zeros(32, 4))
except RuntimeError as e:
    print("RuntimeError:", e)
'''),
    md("""
Read it as: the data arriving is 32 rows by **8** columns, but this layer expects **16**.
The 32 is just the batch size.

**Try it.** Build a 4-32-16-3 network (two hidden layers), print it, and count its weights.
"""),
    code('''
# Try it: your 4-32-16-3 network
'''),
    md("""
## 2. Look at the data first

Samples, features, classes, the range of the values, and the first rows. These decide the
two ends of the network and tell you whether you need to scale.
"""),
    code('''
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits

for load in [load_iris, load_wine, load_breast_cancer, load_digits]:
    d = load()
    X, y = d.data, d.target
    print(f"{load.__name__}: {X.shape[0]} samples, {X.shape[1]} features, {len(d.target_names)} classes, "
          f"values from {X.min():g} to {X.max():g}")
    print("    samples per class:", dict(zip(map(str, d.target_names), np.bincount(y).tolist())))
'''),
    md("**Try it.** Change `load_wine` to any of the other three and look at its first rows."),
    code('''
d = load_wine()
df = pd.DataFrame(d.data, columns=d.feature_names)
df["label"] = d.target
df.head()
'''),
    md("""
## 3. Training on iris and drawing the learning curves

The five-step loop from Topic 14. New around it: `StandardScaler` (fit on the training set
only), `Adam`, and a `history` that records loss and accuracy on both sets after every epoch.
"""),
    code('''
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

torch.manual_seed(0)
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
scaler = StandardScaler().fit(X_train)                 # learn mean and std from the training set only
X_train = torch.tensor(scaler.transform(X_train), dtype=torch.float32)
X_test = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
y_train, y_test = torch.tensor(y_train), torch.tensor(y_test)
loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 3))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}
for epoch in range(200):
    model.train()
    for xb, yb in loader:                              # the same five steps as Topic 14
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
    model.eval()                                       # after every epoch: score both sets
    with torch.no_grad():
        for name, X, y in [("train", X_train, y_train), ("test", X_test, y_test)]:
            out = model(X)
            history[name + "_loss"].append(loss_fn(out, y).item())
            history[name + "_acc"].append((out.argmax(dim=1) == y).float().mean().item())
    if (epoch + 1) in (1, 10, 50, 100, 200):
        print(f"epoch {epoch + 1:3d}  train loss {history['train_loss'][-1]:.3f}"
              f"  test loss {history['test_loss'][-1]:.3f}  test acc {history['test_acc'][-1]:.3f}")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].plot(history["train_loss"], label="train"); ax[0].plot(history["test_loss"], label="test")
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend()
ax[1].plot(history["train_acc"], label="train"); ax[1].plot(history["test_acc"], label="test")
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("accuracy"); ax[1].legend()
plt.show()
'''),
    md("""
Both losses fall together and stay close: a healthy run. The note's run ends at test
accuracy 0.967 (29 of 30 flowers).

**Try it.** Change `range(200)` to `range(500)` and run the cell again. Does the test loss
keep falling?

## 4. Three sizes, built by one function

`hidden` is a list of hidden-layer sizes. `prev` carries Rule 1; `n_in` and `n_out` are Rule 2.
"""),
    code('''
def make_model(n_in, hidden, n_out):
    """[64, 32] -> Linear(n_in, 64), ReLU, Linear(64, 32), ReLU, Linear(32, n_out)"""
    layers, prev = [], n_in
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h                                         # this layer's output is the next one's input
    layers.append(nn.Linear(prev, n_out))                # the last layer: no ReLU
    return nn.Sequential(*layers)


print(make_model(64, [64, 32], 10))
'''),
    md("""
## 5. The experiment: 3 sizes x 3 datasets x 3 seeds

`prepare` splits and scales; `train` is the iris loop and returns the test curve.
Each combination is trained with three seeds and the mean test accuracy is printed.
About half a minute on a CPU.
"""),
    code('''
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepare(X, y, seed):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    scaler = StandardScaler().fit(X_tr)
    f = lambda a: torch.tensor(a, dtype=torch.float32)
    return f(scaler.transform(X_tr)), f(scaler.transform(X_te)), torch.tensor(y_tr), torch.tensor(y_te)


def train(model, X_tr, y_tr, X_te, y_te, epochs=200, seed=0):
    torch.manual_seed(seed)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    test_loss, test_acc = [], []
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            out = model(X_te)
            test_loss.append(loss_fn(out, y_te).item())
            test_acc.append((out.argmax(dim=1) == y_te).float().mean().item())
    return test_loss, test_acc


SIZES = {"Small": [16], "Medium": [64, 32], "Large": [256, 128, 64]}
DATA = {"Wine": load_wine, "Breast cancer": load_breast_cancer, "Digits": load_digits}

for data_name, load in DATA.items():
    X, y = load(return_X_y=True)
    n_in, n_out = X.shape[1], len(np.unique(y))         # sizes come from the data
    for size_name, hidden in SIZES.items():
        accs = []
        for seed in (0, 1, 2):                           # three different splits and starts
            X_tr, X_te, y_tr, y_te = prepare(X, y, seed)
            torch.manual_seed(seed)
            model = make_model(n_in, hidden, n_out)
            test_loss, test_acc = train(model, X_tr, y_tr, X_te, y_te, seed=seed)
            accs.append(test_acc[-1])
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{data_name:14s} {size_name:7s} {n_params:7,d} params   test accuracy {np.mean(accs):.3f}")
'''),
    md("""
## 6. Breast cancer: watch the test loss

Seed 0, the three sizes on one plot. The bigger networks reach their lowest test loss
within a few epochs and then get worse: they are memorizing the 455 training rows.
"""),
    code('''
X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = prepare(X, y, 0)
plt.figure(figsize=(7, 3.8))
for size_name, hidden in SIZES.items():
    torch.manual_seed(0)
    model = make_model(X.shape[1], hidden, 2)
    test_loss, test_acc = train(model, X_tr, y_tr, X_te, y_te, seed=0)
    best = int(np.argmin(test_loss))
    print(f"{size_name:7s} lowest test loss {test_loss[best]:.3f} at epoch {best + 1:3d}, "
          f"at epoch 200 {test_loss[-1]:.3f}")
    plt.plot(test_loss, label=f"{size_name} {hidden}")
plt.xlabel("epoch"); plt.ylabel("test loss"); plt.ylim(0, 0.6); plt.legend()
plt.title("Breast cancer, seed 0: test loss")
plt.show()
'''),
    md("""
## 7. Try it: your own design

Pick a dataset and a list of hidden sizes, and read the curves:

- train accuracy stays low: too small, make it **bigger**;
- train accuracy reaches 1.0 but the test loss climbs: memorizing, make it **smaller**;
- both still improving at the end: **train longer**.
"""),
    code('''
MY_DATA = load_digits          # load_wine, load_breast_cancer or load_digits
MY_HIDDEN = [128, 64]          # your hidden layers
SEED = 0

X, y = MY_DATA(return_X_y=True)
X_tr, X_te, y_tr, y_te = prepare(X, y, SEED)
torch.manual_seed(SEED)
model = make_model(X.shape[1], MY_HIDDEN, len(np.unique(y)))
print(model)
print("weights:", sum(p.numel() for p in model.parameters()))
test_loss, test_acc = train(model, X_tr, y_tr, X_te, y_te, seed=SEED)
print(f"final test accuracy {test_acc[-1]:.3f}, lowest test loss {min(test_loss):.3f} "
      f"at epoch {int(np.argmin(test_loss)) + 1}")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
ax[0].plot(test_loss); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("test loss")
ax[1].plot(test_acc); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("test accuracy")
plt.show()
'''),
    md(f"""
## What to take away

The ends of a network come from the data; the middle is your choice. Start small, plot the
curves, and grow the network only when the data has more to learn and enough rows to learn
it from. Back to the [lecture note]({NOTE15}).
"""),
]

if __name__ == "__main__":
    write("topic14_pytorch_essentials.ipynb", T14)
    write("topic15_designing_networks.ipynb", T15)
