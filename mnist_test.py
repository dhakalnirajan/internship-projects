import os
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from nn.activations import relu, softmax
from nn.autograd import Tensor
from nn.layers import Conv2D, Dense, Dropout, Flatten
from nn.losses import CrossEntropy
from nn.models import Sequential
from nn.optimizers import Adam

MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz"  # Note: Standard numpy format via mirror or alternative if needed, keeping your original S3 link below:
MNIST_URL = "https://s3.amazonaws.com/img-datasets/mnist.npz"
MNIST_FILE = "mnist.npz"

if not os.path.exists(MNIST_FILE):
    print("Downloading MNIST...")
    with urllib.request.urlopen(MNIST_URL) as response:
        with open(MNIST_FILE, "wb") as f:
            f.write(response.read())
    print("Done.")

data = np.load(MNIST_FILE)

X_train = data["x_train"].astype(np.float32) / 255.0
y_train = data["y_train"].astype(np.int32)

X_test = data["x_test"].astype(np.float32) / 255.0
y_test = data["y_test"].astype(np.int32)

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

y_train = np.eye(10, dtype=np.float32)[y_train]
y_test_onehot = np.eye(10, dtype=np.float32)[y_test]

model = Sequential()

model.add(
    Conv2D(
        filters=32,
        kernel_size=3,
        strides=1,
        padding="valid",
    )
)

model.add(
    Conv2D(
        filters=64,
        kernel_size=3,
        strides=1,
        padding="valid",
    )
)

model.add(Flatten())

model.add(
    Dense(
        128,
        activation=relu,
    )
)

model.add(Dropout(0.5))

model.add(
    Dense(
        10,
        activation=softmax,
    )
)

dummy = np.random.randn(1, 28, 28, 1)
model(dummy)

optimizer = Adam(
    model.parameters(),
    lr=0.001,
)

model.compile(
    optimizer=optimizer,
    loss=CrossEntropy(),
)

epochs = 10
batch_size = 128

num_train = X_train.shape[0]

loss_history = []
accuracy_history = []

for epoch in range(epochs):
    permutation = np.random.permutation(num_train)

    X_epoch = X_train[permutation]
    y_epoch = y_train[permutation]

    epoch_loss = 0.0

    for start in range(0, num_train, batch_size):
        end = min(start + batch_size, num_train)

        x = Tensor(
            X_epoch[start:end],
            requires_grad=False,
        )

        y = Tensor(
            y_epoch[start:end],
            requires_grad=False,
        )

        prediction = model.forward(
            x,
            training=True,
        )

        loss = model.loss.forward(
            y,
            prediction,
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        epoch_loss += loss.data.item() * (end - start)

    epoch_loss /= num_train

    prediction = model.predict(X_test)

    predicted = np.argmax(
        prediction.data,
        axis=1,
    )

    accuracy = np.mean(predicted == y_test)

    loss_history.append(epoch_loss)
    accuracy_history.append(accuracy)

    print(
        f"Epoch {epoch + 1:02d}/{epochs} "
        f"Loss: {epoch_loss:.6f} "
        f"Accuracy: {accuracy:.4f}"
    )

prediction = model.predict(X_test)

predicted = np.argmax(
    prediction.data,
    axis=1,
)

accuracy = np.mean(predicted == y_test)

print()
print(f"Final Accuracy: {accuracy:.4f}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.plot(
    range(1, epochs + 1),
    loss_history,
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)

plt.subplot(1, 2, 2)

plt.plot(
    range(1, epochs + 1),
    accuracy_history,
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Test Accuracy")
plt.grid(True)

plt.tight_layout()
plt.savefig("mnist_training_curves.png")
plt.show()

cm = confusion_matrix(
    y_test,
    predicted,
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
)

disp.plot()

plt.title("Confusion Matrix")
plt.savefig("mnist_confusion_matrix.png")
plt.show()