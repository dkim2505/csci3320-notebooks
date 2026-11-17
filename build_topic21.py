"""Build the companion notebook for CSCI 3320 Topic 21 from the lecture note.

    python3 build_topic21.py

Unlike the earlier topics, this notebook carries the whole note, not only its code:
the explanations, the tables and the pictures sit between the code cells, in the
note's order, so it can be read on its own. Read straight from the homepage repo:

    ../homepage/src/content/csci3320/session23.md

- ```python blocks become code cells, unchanged.
- ```text blocks (the output recorded for the note, on an A100) become a short
  "output in the note" cell under the code, to compare with.
- Pictures and links that start with / point at https://dongchul.kim.
- :::prereq and :::note become quotes; :::hint (the answer) folds away.
- A new markdown cell starts at every ## or ### heading.

The notebook asks Colab for a T4 GPU.
"""
import json
import re
from build_topic08_12 import NOTES, META, SITE, md, code

KEY, TOPIC, NAME = "23", "21", "topic21_transfer_learning"
TITLE = "Transfer Learning: Adapting a Pretrained CNN to Your Own Images"
HOST = "https://dongchul.kim"
DROP = "PyTorch does not run in the page"      # the note's own "run it in Colab" paragraph


def markdown(text):
    """The note's markdown, made to work in Colab."""
    text = re.sub(r"\]\(/", f"]({HOST}/", text)
    text = re.sub(r"^:::prereq\n(.*?)^:::\s*$", lambda m: quote("Before this topic", m.group(1)), text, flags=re.S | re.M)
    text = re.sub(r"^:::note\n(.*?)^:::\s*$", lambda m: quote("Note", m.group(1)), text, flags=re.S | re.M)
    text = re.sub(r"^:::hint\n(.*?)^:::\s*$",
                  lambda m: f"<details><summary><b>Show the answer</b></summary>\n\n{m.group(1).strip()}\n\n</details>\n",
                  text, flags=re.S | re.M)
    assert ":::" not in text, "a ::: block this builder does not know"
    return text


def quote(label, body):
    lines = body.strip().splitlines()
    return "\n".join([f"> **{label}.**", ">"] + [f"> {l}" if l else ">" for l in lines]) + "\n"


def split_at_headings(text):
    parts = re.split(r"(?m)^(?=#{2,3} )", text)
    return [p.strip() for p in parts if p.strip()]


def build():
    note = (NOTES / f"session{KEY}.md").read_text()
    lines = note.splitlines(True)
    assert lines[0].startswith("<div"), "the note no longer starts with the Colab box"
    note = "".join(lines[1:])
    note = "\n\n".join(p for p in note.split("\n\n") if not p.strip().startswith(DROP))

    url = SITE + KEY
    cells = [md(f"""
# Topic {TOPIC}: {TITLE}

CSCI 3320 · AI Programming in Python · Fall 2026

This notebook is the [Topic {TOPIC} lecture note]({url}): the same explanations and pictures,
with every code cell in order. Run the cells from top to bottom (Runtime > Run all works too),
then change things and run them again. Under some code cells, "Output in the note" shows what
that cell printed when the note was made, so you can compare.

**Use a GPU for this one**: Runtime > Change runtime type > T4 GPU (free). Way 2,
fine-tuning, takes roughly 5 to 10 minutes on a T4 and hours on a CPU. Numbers can differ
a little from the note on a different GPU or library version.
""")]
    fence = re.compile(r"^```(\w*)[^\n]*\n(.*?)^```\s*$", re.S | re.M)
    pos, n_code = 0, 0
    for m in fence.finditer(note):
        for part in split_at_headings(markdown(note[pos:m.start()])):
            cells.append(md(part))
        pos = m.end()
        lang, body = m.group(1), m.group(2)
        if lang == "python":
            cells.append(code(body)); n_code += 1
        elif lang == "text":
            cells.append(md("*Output in the note:*\n\n```text\n" + body.rstrip("\n") + "\n```"))
        else:
            cells.append(md(f"```{lang}\n{body}```"))
    for part in split_at_headings(markdown(note[pos:])):
        cells.append(md(part))
    cells.append(md(f"Back to the [Topic {TOPIC} lecture note]({url})."))

    meta = json.loads(json.dumps(META))
    meta["accelerator"] = "GPU"
    meta["colab"]["gpuType"] = "T4"
    with open(f"{NAME}.ipynb", "w") as f:
        json.dump({"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 0}, f, indent=1)
    n_md = sum(c["cell_type"] == "markdown" for c in cells)
    print(f"wrote {NAME}.ipynb  ({n_code} code cells, {n_md} text cells), asks for a T4 GPU")


if __name__ == "__main__":
    build()
