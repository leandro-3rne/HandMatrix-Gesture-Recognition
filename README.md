# HandMatrix: Gesture Recognition Engine ✋🤖

A comprehensive Computer Vision project implemented in C++ and Python to recognize hand gestures in real-time. This project explores two different approaches: a **Custom Neural Network (MLP)** built from scratch in C++ and a **Convolutional Neural Network (CNN)** trained in Python and deployed in C++.

![C++](https://img.shields.io/badge/C++-20-blue.svg) ![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg) ![Python](https://img.shields.io/badge/Python-3.12-yellow.svg) ![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange.svg)

## 📂 Project Overview

This repository contains four distinct modules:
1.  **Data Collector (C++):** A tool to capture and label grayscale hand gestures.
2.  **Data Augmentation & Training (Python):** Scripts to augment data and train a CNN (exporting to ONNX).
3.  **Custom NN Inference (C++):** A Multilayer Perceptron implemented purely in C++ (Eigen library) with manual backpropagation.
4.  **CNN Inference (C++):** Live inference using the Python-trained model via OpenCV DNN module.

---

## 🧠 Theoretical Background

This project compares two fundamental architectures in Deep Learning.

### 1. Multilayer Perceptron (Custom C++ Implementation)
The custom neural network processes the image as a flat vector. The image ($32 \times 32$ pixels) is flattened into a vector $x$ of size 1024.

**Forward Propagation:**
For a layer $l$, the activation $a^{[l]}$ is calculated as:

$$z^{[l]} = W^{[l]} \cdot a^{[l-1]} + b^{[l]}$$

$$a^{[l]} = \sigma(z^{[l]})$$

Where:
* $W$ is the Weight Matrix.
* $b$ is the Bias Vector.
* $\sigma$ is the Sigmoid Activation Function: $\sigma(z) = \frac{1}{1 + e^{-z}}$.

**Backpropagation (Learning):**
The network minimizes the error by computing the gradient of the Cost Function w.r.t. weights using the Chain Rule:

$$\frac{\partial C}{\partial W} = \frac{\partial C}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial W}$$

### 2. Convolutional Neural Network (CNN)
Unlike the MLP, the CNN preserves the spatial structure of the image ($32 \times 32 \times 1$). It is translation invariant, making it superior for image recognition.

**Convolution Operation:**
The core operation involves a kernel $K$ sliding over the input image $I$:

$$(I * K)(i, j) = \sum_m \sum_n I(i+m, j+n) \cdot K(m, n)$$

**Feature Extraction Pipeline:**
1.  **Conv2D:** Detects edges and patterns.
2.  **ReLU:** Introduces non-linearity ($f(x) = \max(0, x)$).
3.  **MaxPooling:** Reduces dimensionality (Downsampling).
4.  **Flatten & Dense:** Final classification.

---

## 🚀 How to Use (Workflow)

To reproduce the results or train with your own data, follow this strict order:

### Step 1: Data Collection 📸
* Go to `01_Data_Collector`.
* Compile and run the C++ program.
* Press **'B'** to capture the background (ensure no hand is in the frame).
* Select a class (0-7) using number keys.
* Hold **SPACE** to record images. They are saved into labeled subfolders automatically.

### Step 2: Augmentation & Training 🐍
* Go to `02_Data_Augmenter`.
* Run `augment_data.py` to generate variations (rotations, noise) of your raw data.
* Run `train_cnn.py`. This uses TensorFlow/Keras to train the CNN.
* **Result:** A file named `hand_cnn.onnx` will be generated.

### Step 3: Deployment (Inference) ⚡

**Option A: The CNN (Recommended)**
* Go to `04_Model_CNN_Inference`.
* Copy the `hand_cnn.onnx` file into the build directory (next to the executable).
* Run the program. It uses OpenCV's DNN module to load the ONNX file.

**Option B: The Custom NN**
* Go to `03_Model_CustomNN`.
* This module reads the augmented data directly from the folders and performs training in C++ (Backpropagation) before switching to live inference mode.

---

## 🛠 Dependencies

* **C++:** OpenCV 4.x, Eigen3 (for Custom NN)
* **Python:** TensorFlow, NumPy < 2.0, tf2onnx, OpenCV-Python, tf_keras

## 📝 License
This project is open-source. Feel free to use and modify.
