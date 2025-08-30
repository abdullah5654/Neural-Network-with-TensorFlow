# README - Simple Neural Network with TensorFlow (MNIST Classification)

## 📌 Overview
This project demonstrates how to build, train, and evaluate simple neural networks using **TensorFlow/Keras** on the **MNIST dataset** (handwritten digit classification). It includes two different architectures: a **Multi-Layer Perceptron (MLP)** and a **Convolutional Neural Network (CNN)**, allowing comparison of performance between the two approaches.

---

## 🎯 Objectives
- Understand the fundamentals of deep learning model development.
- Learn the workflow of dataset preprocessing, model creation, training, and evaluation.
- Visualize training progress and test predictions.
- Compare performance of MLP vs CNN models.

---

## 🛠 Features
1. **Dataset Preparation**
   - Uses the MNIST dataset (28x28 grayscale images of digits 0–9).
   - Normalizes pixel values to the range `[0, 1]`.
   - Prepares flattened data for MLP and reshaped data for CNN.

2. **Model Architectures**
   - **MLP Model:** Fully connected layers with dropout for regularization.
   - **CNN Model:** Convolution + pooling layers followed by dense layers.

3. **Training & Evaluation**
   - Optimizer: Adam
   - Loss: Categorical Crossentropy
   - Metrics: Accuracy
   - Evaluates performance on the test set.

4. **Visualization**
   - Plots training/validation accuracy and loss over epochs.
   - Displays confusion matrix for CNN predictions.
   - Shows example test images with predicted labels.

5. **Saving & Loading**
   - Trained CNN model is saved in `.h5` format.
   - Model can be reloaded and re-evaluated.

---

## 📊 Results
- MLP and CNN both classify handwritten digits, but **CNN generally achieves higher accuracy** due to its ability to capture spatial features.
- Accuracy, precision, recall, and F1-scores are printed for the CNN model.

---

## 🚀 How to Run
1. Install dependencies:
   ```bash
   pip install tensorflow matplotlib seaborn scikit-learn
   ```

2. Run the script in a Python environment (Jupyter, Colab, or terminal).

3. View results:
   - Training/validation accuracy & loss plots.
   - Confusion matrix.
   - Sample predictions.

4. Load trained model:
   ```python
   loaded_model = tf.keras.models.load_model("cnn_mnist_model.h5")
   ```

---

## 💡 Best Practices & Extensions
- Use GPU for faster training.
- Experiment with different optimizers (SGD, RMSprop).
- Add batch normalization or dropout for better generalization.
- Try deeper CNNs or RNNs for more complex datasets.
- Replace MNIST with CIFAR-10 or a text dataset (IMDb) for further experimentation.

---

## 📂 Project Contents
- `main.py` → Full implementation code
- `cnn_mnist_model.h5` → Saved trained CNN model (after running script)

---

## ✅ Learning Outcomes
By completing this project, you will:
- Understand the structure of neural networks.
- Gain practical experience in building and training models.
- Learn how to evaluate and visualize deep learning results.
- Be able to extend this workflow to other datasets and model architectures.
