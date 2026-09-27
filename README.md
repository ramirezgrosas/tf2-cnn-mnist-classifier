# CNN Classifier for MNIST (TensorFlow 2)

Convolutional neural network that classifies handwritten digits from the
[MNIST dataset](http://yann.lecun.com/exdb/mnist/), built with TensorFlow 2 / Keras.

This project is based on the Week 2 programming assignment of the
*Getting started with TensorFlow 2* course.

## Goal

Build, compile and train a CNN on MNIST (60,000 training and 10,000 test images of
28x28 grayscale digits) and evaluate its accuracy on the test set.

## Results

| Metric   | Training (epoch 5) | Test       |
|----------|--------------------|------------|
| Accuracy | 98.89%             | **98.28%** |
| Loss     | 0.0341             | **0.0604** |

The model reaches **98.28% accuracy on the 10,000 test images**, around 17 mistakes for every
1000 digits. Training was stable: accuracy went from 93.22% in the first epoch to 98.89% in the
fifth, and the loss dropped from 0.2272 to 0.0341, at roughly 3-4 seconds per epoch on CPU.

The 0.6 point gap between training and test accuracy points to only mild overfitting, so the
network generalises well to images it never saw.

## Model

A small `Sequential` network with 105,306 trainable parameters:

| Layer          | Configuration                        | Output shape  |
|----------------|--------------------------------------|---------------|
| `Conv2D`       | 8 filters 3x3, `padding="SAME"`, ReLU | (28, 28, 8)  |
| `MaxPooling2D` | 2x2                                  | (14, 14, 8)   |
| `Flatten`      | -                                    | (1568,)       |
| `Dense`        | 64 units, ReLU                       | (64,)         |
| `Dense`        | 64 units, ReLU                       | (64,)         |
| `Dense`        | 10 units, softmax                    | (10,)         |

Trained for 5 epochs with the Adam optimizer and `sparse_categorical_crossentropy` as the loss
function (the labels are integers, not one-hot vectors).

## Notebook contents

`mnist_cnn_classifier.ipynb` follows the full workflow:

1. **Load and preprocess the data** - pixel values scaled to the `[0, 1]` range and a channel
   dimension added, so each image becomes a `(28, 28, 1)` tensor.
2. **Build the model** - one convolutional block plus a small dense classifier.
3. **Compile and train** - Adam, sparse categorical cross entropy, 5 epochs.
4. **Learning curves** - accuracy and loss per epoch, plotted from the training history.
5. **Evaluation** - loss and accuracy on the test set.
6. **Predictions** - four random test digits shown next to the probability distribution the model
   assigns to each class.

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
git clone https://github.com/ramirezgrosas/tf2-cnn-mnist-classifier.git
cd tf2-cnn-mnist-classifier
uv sync
```

Then open `mnist_cnn_classifier.ipynb` and select the `.venv` environment as the kernel.

The dataset is downloaded automatically by Keras the first time the notebook runs, so there is
nothing else to set up.

## Possible improvements

- Use a validation split during training to monitor generalisation epoch by epoch.
- Add more convolutional blocks with more filters, the usual way to push MNIST above 99%.
- Add regularisation (dropout or batch normalisation) when training for more epochs.
- Build a confusion matrix to find out which digits get confused with each other.
