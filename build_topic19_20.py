"""Build the companion notebooks for CSCI 3320 Topics 19 and 20 from the lecture notes.

    python3 build_topic19_20.py
"""
from build_topic08_12 import build

TOPICS = [
    ("21", "19", "topic19_cnn_filters_sizes_pooling", "Convolutional Neural Networks: Filters, Feature Maps, Sizes, and Pooling"),
    ("22", "20", "topic20_training_a_cnn", "Training a CNN: MNIST, Fashion-MNIST, and the MLP Compared"),
]

if __name__ == "__main__":
    for t in TOPICS:
        build(*t)
