import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# -----------------------------
# 1. Dataset Preparation
# -----------------------------

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values (0-1)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Flatten for MLP
x_train_flat = x_train.reshape(-1, 28*28)
x_test_flat = x_test.reshape(-1, 28*28)

# One-hot encode labels
y_train_cat = tf.keras.utils.to_categorical(y_train, 10)
y_test_cat = tf.keras.utils.to_categorical(y_test, 10)

# -----------------------------
# 2. Model Architectures
# -----------------------------

# --- MLP Model ---
mlp_model = models.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

mlp_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# --- CNN Model ---
cnn_model = models.Sequential([
    layers.Input(shape=(28,28,1)),
    layers.Conv2D(32, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax")
])

cnn_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Reshape input for CNN
x_train_cnn = np.expand_dims(x_train, -1)
x_test_cnn = np.expand_dims(x_test, -1)

# -----------------------------
# 3. Training & Evaluation
# -----------------------------

# Train MLP
history_mlp = mlp_model.fit(x_train_flat, y_train_cat, validation_split=0.2, epochs=5, batch_size=128, verbose=2)

# Train CNN
history_cnn = cnn_model.fit(x_train_cnn, y_train_cat, validation_split=0.2, epochs=5, batch_size=128, verbose=2)

# Evaluate
mlp_eval = mlp_model.evaluate(x_test_flat, y_test_cat, verbose=0)
cnn_eval = cnn_model.evaluate(x_test_cnn, y_test_cat, verbose=0)

print("\nMLP Test Accuracy:", mlp_eval[1])
print("CNN Test Accuracy:", cnn_eval[1])

# -----------------------------
# 4. Metrics & Visualization
# -----------------------------

# Plot training curves
def plot_history(history, title):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(title + ' Accuracy')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(title + ' Loss')
    plt.legend()
    plt.show()

plot_history(history_mlp, "MLP")
plot_history(history_cnn, "CNN")

# Classification report for CNN
y_pred_cnn = np.argmax(cnn_model.predict(x_test_cnn), axis=1)
print(classification_report(y_test, y_pred_cnn))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_cnn)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - CNN")
plt.show()

# Show sample predictions
for i in range(5):
    plt.imshow(x_test[i], cmap="gray")
    plt.title(f"True: {y_test[i]}, Pred: {y_pred_cnn[i]}")
    plt.show()

# -----------------------------
# 5. Save & Load Model
# -----------------------------

cnn_model.save("cnn_mnist_model.h5")
loaded_model = tf.keras.models.load_model("cnn_mnist_model.h5")
print("Loaded model test accuracy:", loaded_model.evaluate(x_test_cnn, y_test_cat, verbose=0)[1])
