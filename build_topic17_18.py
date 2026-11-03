"""Build the companion notebooks for CSCI 3320 Topics 17 and 18 from the lecture notes.

Same as build_topic08_12.py: the notes run their code in the page (Pyodide), and each
notebook is every ```python block of the note, in order and unchanged, under the
note's own section headings. Read straight from the homepage repo:

    ../homepage/src/content/csci3320/session18.md    (Topic 17, decision trees)
    ../homepage/src/content/csci3320/session18b.md   (Topic 18, Naive Bayes)

    python3 build_topic17_18.py
"""
from build_topic08_12 import build

TOPICS = [
    ("18", "17", "topic17_decision_trees",
     "Decision Trees: Entropy, Information Gain, and Why One Tree Overfits"),
    ("18b", "18", "topic18_naive_bayes",
     "Naive Bayes: Probability by Counting, Bayes' Rule, and Gaussian Naive Bayes"),
]

if __name__ == "__main__":
    for t in TOPICS:
        build(*t)
