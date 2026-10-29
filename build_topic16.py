"""Build the companion notebook for CSCI 3320 Topic 16 (overfitting and regularization).

Same recipe as build_topic08_12.py: every ```python block of the lecture note, in
order, under the note's own section headings. One difference: a block that is an
excerpt (it contains a line that is only "...") is left out, because the full
code is already in an earlier cell.

    python3 build_topic16.py
"""
import build_topic08_12 as b

_blocks = b.blocks


def blocks(text):
    for item in _blocks(text):
        if item[0] == "code" and any(line.strip() == "..." for line in item[1].splitlines()):
            continue
        yield item


b.blocks = blocks

if __name__ == "__main__":
    b.build("15", "16", "topic16_regularization", "Overfitting and Regularization: L2 Weight Decay, Dropout, and Early Stopping")
