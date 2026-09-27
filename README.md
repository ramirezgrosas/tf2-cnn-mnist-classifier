# CNN Classifier for MNIST (TensorFlow 2)

Convolutional neural network that classifies handwritten digits from the
[MNIST dataset](http://yann.lecun.com/exdb/mnist/), built with TensorFlow 2 / Keras.

This project is based on the Week 2 programming assignment of the
*Getting started with TensorFlow 2* course.

> 🚧 **Work in progress.** Results and analysis will be added as the project advances.

## Goal

Build, compile and train a CNN on MNIST (60,000 training and 10,000 test images of
28x28 grayscale digits) and evaluate its accuracy on the test set.

## Project structure

```
tf2-cnn-mnist-classifier/
├── mnist_cnn_classifier.ipynb   # main notebook: data, model, training, evaluation
├── data/                        # images used in the notebook
├── src/                         # Python package created by uv
├── pyproject.toml               # project metadata and dependencies
└── uv.lock                      # exact dependency versions
```

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/<your-user>/tf2-cnn-mnist-classifier.git
cd tf2-cnn-mnist-classifier
uv sync
```

Then open `mnist_cnn_classifier.ipynb` and select the `.venv` environment as the kernel.

## Results

_Coming soon._
