"""Build the CSCI 3320 Topic 23 (Transformers, Part II) Colab notebook.

    python3 build_topic23.py

It is CSCI 6379's Topic 32 notebook (../csci6379-notebooks/topic32_transformer_translation.ipynb,
itself built from the slides' code by build_topic32.py there), reused as it is (user, 2026-11-18).
Only the opening cell changes: this course's topic number, note and slides.
"""
import json
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "csci6379-notebooks/topic32_transformer_translation.ipynb"
NAME = "topic23_transformer_translation"
OLD_HEAD = ("# Topic 32 · A Transformer from scratch: English → Spanish\n\n"
            "CSCI 6379 · companion to [Transformers, Part II: the decoder](https://dongchul.kim/csci6379-fall2026/topic/32) "
            "and its [slides](https://dongchul.kim/csci6379/slides/topic32.pdf) (the Implementation section).")
NEW_HEAD = ("# Topic 23 · A Transformer from scratch: English → Spanish\n\n"
            "CSCI 3320 · AI Programming in Python · Fall 2026 · companion to "
            "[Transformers, Part II: the decoder](https://dongchul.kim/csci3320-fall2026/topic/25) "
            "and its [slides](https://dongchul.kim/csci3320/slides-pdf/session25.pdf) (the Implementation section).")

nb = json.loads(SRC.read_text())
head = "".join(nb["cells"][0]["source"])
assert head.startswith(OLD_HEAD), "the 6379 notebook's opening cell changed"
nb["cells"][0]["source"] = (NEW_HEAD + head[len(OLD_HEAD):]).splitlines(True)
text = json.dumps(nb)
assert "csci6379" not in text and "6379" not in text, "the notebook still names the other course"
Path(f"{NAME}.ipynb").write_text(json.dumps(nb, indent=1))
print(f"wrote {NAME}.ipynb ({sum(c['cell_type'] == 'code' for c in nb['cells'])} code cells)")
