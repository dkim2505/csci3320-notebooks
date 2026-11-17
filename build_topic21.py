"""Build the companion notebook for CSCI 3320 Topic 21 from the lecture note.

    python3 build_topic21.py

Same builder as the other topics, but this notebook asks for a GPU: Colab opens it
on a T4 runtime, and the opening cell says so.
"""
import json
from build_topic08_12 import build

NAME = "topic21_transfer_learning"
CPU_LINE = "It runs on a CPU; no GPU is needed."
GPU_LINE = ("**Use a GPU for this one**: Runtime > Change runtime type > T4 GPU (free). Way 2,\n"
            "fine-tuning, takes roughly 5 to 10 minutes on a T4 and hours on a CPU.")

if __name__ == "__main__":
    build("23", "21", NAME, "Transfer Learning: Adapting a Pretrained CNN to Your Own Images")
    nb = json.load(open(f"{NAME}.ipynb"))
    intro = "".join(nb["cells"][0]["source"])
    assert CPU_LINE in intro, "intro text changed in build_topic08_12.py"
    intro = intro.replace(CPU_LINE, GPU_LINE)
    nb["cells"][0]["source"] = intro.splitlines(True)
    nb["metadata"]["accelerator"] = "GPU"
    nb["metadata"]["colab"]["gpuType"] = "T4"
    json.dump(nb, open(f"{NAME}.ipynb", "w"), indent=1)
    print("asks for a T4 GPU")
