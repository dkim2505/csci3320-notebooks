"""Build the companion notebooks for CSCI 3320 Topics 8a to 12 from the lecture notes.

The notes for these topics run their code in the page (Pyodide). Each notebook here is
that same code for Colab: every ```python block of the note, in order and unchanged,
under the note's own section headings. Read straight from the homepage repo, so a
rebuild picks up any later edit to a note:

    ../homepage/src/content/csci3320/session<key>.md

Only one cell is replaced: Topic 11a's "copy students.csv into the browser" cell, which
uses pyodide.http. In Colab the same cell downloads the two data files instead.

    python3 build_topic08_12.py
"""
import json
import re
from pathlib import Path

NOTES = Path(__file__).resolve().parent.parent / "homepage/src/content/csci3320"
SITE = "https://dongchul.kim/csci3320-fall2026/topic/"

META = {"colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}

# note key -> (topic number students see, file name, title)
TOPICS = [
    ("8", "8a", "topic08a_linear_regression_gradient_descent", "Linear Regression, MSE, and Gradient Descent"),
    ("8b", "8b", "topic08b_linear_regression_in_practice", "Linear Regression in Practice: scikit-learn, RMSE, and R squared"),
    ("9", "9a", "topic09a_linear_classification", "Linear Classification, and How to Judge a Classifier"),
    ("9b", "9b", "topic09b_logistic_regression_in_practice", "Logistic Regression in Practice: Real Patients, by Hand and with scikit-learn"),
    ("9c", "10", "topic10_learning_curves", "Learning Curves: Watching Logistic Regression Learn"),
    ("11a", "11a", "topic11a_ml_in_code", "Machine Learning in Code: Data, Model, Training, Evaluation"),
    ("10", "11b", "topic11b_softmax", "Softmax: From Two Classes to Many, by Hand, in scikit-learn, and in PyTorch"),
    ("11", "12", "topic12_cross_validation", "Cross-Validation: Why One Split Is Not Enough, and Why One CV Is Not Either"),
]

# Cells that only make sense in the browser, replaced by their Colab equivalent.
REPLACE = {
    "from pyodide.http import open_url": '''
# In Colab: download the two data files from the course website into this notebook's folder.
import urllib.request
for name in ["students.csv", "students.xlsx"]:
    urllib.request.urlretrieve("https://dongchul.kim/csci3320/data/" + name, name)
print("saved students.csv and students.xlsx")
''',
}


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(True)}


def blocks(text):
    """Yield ("heading", level, title) and ("code", source) in the order of the note."""
    fence = re.compile(r"^```(\w*)[^\n]*\n(.*?)^```\s*$", re.S | re.M)
    pos = 0
    for m in fence.finditer(text):
        for h in re.finditer(r"^(#{2,3}) (.+)$", text[pos:m.start()], re.M):
            yield ("heading", len(h.group(1)), h.group(2).strip())
        pos = m.end()
        if m.group(1) == "python":
            src = m.group(2)
            src = re.sub(r"\A#@run[^\n]*\n", "", src)
            for key, new in REPLACE.items():
                if key in src:
                    src = new
            yield ("code", src)


def build(key, topic_no, name, title):
    text = (NOTES / f"session{key}.md").read_text()
    url = SITE + key
    cells = [md(f"""
# Topic {topic_no}: {title}

CSCI 3320 · AI Programming in Python · Fall 2026

This notebook is the code from the [Topic {topic_no} lecture note]({url}), every cell in the
same order, under the note's own section headings. The explanations are in the note: keep it
open beside this notebook. Run the cells from top to bottom (Runtime > Run all works too),
then change things and run them again.

It runs on a CPU; no GPU is needed. Numbers can differ a little from the note on a
different library version.
""")]
    pending = []          # headings since the last code cell
    n_code = 0
    for item in blocks(text):
        if item[0] == "heading":
            pending.append(item)
            continue
        if pending:
            # keep the section (##) and, under it, the latest sub-section (###)
            top = [h for h in pending if h[1] == 2]
            sub = [h for h in pending if h[1] == 3]
            lines = []
            if top:
                lines.append("## " + top[-1][2])
                sub = [h for h in pending[pending.index(top[-1]):] if h[1] == 3]
            if sub:
                lines.append("### " + sub[-1][2])
            cells.append(md("\n\n".join(lines)))
            pending = []
        cells.append(code(item[1]))
        n_code += 1
    cells.append(md(f"Back to the [Topic {topic_no} lecture note]({url})."))
    with open(f"{name}.ipynb", "w") as f:
        json.dump({"cells": cells, "metadata": META, "nbformat": 4, "nbformat_minor": 0}, f, indent=1)
    print(f"wrote {name}.ipynb  ({n_code} code cells)")


if __name__ == "__main__":
    for t in TOPICS:
        build(*t)
