# CNN with MNIST and CIFAR-10 on Google Colab

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.x-brightgreen.svg)
![PyTorch](https://img.shields.io/badge/pytorch-1.x-brightgreen.svg)

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Datasets](#datasets)
  - [MNIST](#mnist)
  - [CIFAR-10](#cifar-10)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
- [Usage](#usage)
  - [Running on Google Colab](#running-on-google-colab)
  - [Training the CNN with TensorFlow](#training-the-cnn-with-tensorflow)
    - [For MNIST](#for-mnist-tensorflow)
    - [For CIFAR-10](#for-cifar-10-tensorflow)
  - [Training the CNN with PyTorch](#training-the-cnn-with-pytorch)
    - [For MNIST](#for-mnist-pytorch)
    - [For CIFAR-10](#for-cifar-10-pytorch)
  - [Evaluating the Model](#evaluating-the-model)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction
This repository provides comprehensive guides to building and training Convolutional Neural Networks (CNNs) using the MNIST and CIFAR-10 datasets on Google Colab. It supports both **TensorFlow** and **PyTorch**, allowing users to choose their preferred deep learning framework. Leveraging Google Colab's free GPU resources, you can efficiently train models without the need for local computational resources.

## Features
- **Easy-to-follow Jupyter Notebooks**: Step-by-step instructions to build CNNs in TensorFlow and PyTorch.
- **Support for MNIST and CIFAR-10**: Two widely-used datasets for image classification.
- **Google Colab Integration**: Seamless setup and execution on Colab.
- **Visualizations**: Plot training metrics and sample predictions.
- **Modular Code**: Easily adaptable for other datasets and model architectures.

## Datasets

### MNIST
- **Description**: A dataset of 70,000 handwritten digits (0-9) with 60,000 training and 10,000 testing samples.
- **Image Size**: 28x28 grayscale images.
- **Use Case**: Ideal for beginners to understand image classification and CNNs.

### CIFAR-10
- **Description**: A dataset containing 60,000 32x32 color images across 10 classes, with 50,000 training and 10,000 testing samples.
- **Image Size**: 32x32 RGB images.
- **Use Case**: Suitable for more complex image classification tasks compared to MNIST.

## Getting Started

### Prerequisites
- **Google Account**: To access Google Colab.
- **Web Browser**: Latest version of Chrome, Firefox, or Safari.

### Setup
1. **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/cnn-mnist-cifar10-colab.git
    ```
2. **Navigate to the Project Directory**
    ```bash
    cd cnn-mnist-cifar10-colab
    ```
3. **Open the Notebook in Google Colab**
    - Go to [Google Colab](https://colab.research.google.com/).
    - Click on `File` > `Upload notebook`.
    - Upload the desired notebook:
      - `tensorflow/mnist_cnn_tf.ipynb` or `tensorflow/cifar10_cnn_tf.ipynb` for TensorFlow.
      - `pytorch/mnist_cnn_pt.ipynb` or `pytorch/cifar10_cnn_pt.ipynb` for PyTorch.

## Usage

### Running on Google Colab
1. **Ensure GPU is Enabled**
    - In Colab, go to `Runtime` > `Change runtime type`.
    - Select `GPU` from the Hardware accelerator dropdown.
    - Click `Save`.

2. **Install Required Libraries**
    - **TensorFlow**
        ```python
        !pip install tensorflow matplotlib
        ```
    - **PyTorch**
        ```python
        !pip install torch torchvision matplotlib
        ```

3. **Import Libraries**
    - **TensorFlow**
        ```python
        import tensorflow as tf
        from tensorflow.keras import datasets, layers, models
        import matplotlib.pyplot as plt
        ```
    - **PyTorch**
        ```python
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torch.optim as optim
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader
        import matplotlib.pyplot as plt
        ```

### Training the CNN with TensorFlow

#### For MNIST (TensorFlow)
1. **Load the Dataset**
    ```python
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
    test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
    ```

2. **Build the Model**
    ```python
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    ```

3. **Compile the Model**
    ```python
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    ```

4. **Train the Model**
    ```python
    history = model.fit(train_images, train_labels, epochs=10, 
                        validation_data=(test_images, test_labels))
    ```

#### For CIFAR-10 (TensorFlow)
1. **Load the Dataset**
    ```python
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    ```

2. **Build the Model**
    ```python
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    ```

3. **Compile the Model**
    ```python
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    ```

4. **Train the Model**
    ```python
    history = model.fit(train_images, train_labels, epochs=20, 
                        validation_data=(test_images, test_labels))
    ```

### Training the CNN with PyTorch

#### For MNIST (PyTorch)
1. **Load the Dataset**
    ```python
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    ```

2. **Define the Model**
    ```python
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)

    model = CNN().to(device)
    ```

3. **Define Loss and Optimizer**
    ```python
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    ```

4. **Train the Model**
    ```python
    for epoch in range(10):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')
    ```

#### For CIFAR-10 (PyTorch)
1. **Load the Dataset**
    ```python
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    ```

2. **Define the Model**
    ```python
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.conv3 = nn.Conv2d(64, 128, 3, 1)
            self.fc1 = nn.Linear(128 * 2 * 2, 256)
            self.fc2 = nn.Linear(256, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv3(x))
            x = F.max_pool2d(x, 2)
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)

    model = CNN().to(device)
    ```

3. **Define Loss and Optimizer**
    ```python
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    ```

4. **Train the Model**
    ```python
    for epoch in range(20):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')
    ```

### Evaluating the Model
1. **Evaluate on Test Data**
    - **TensorFlow**
        ```python
        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
        print(f'Test accuracy: {test_acc}')
        ```
    - **PyTorch**
        ```python
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        test_accuracy = correct / len(test_loader.dataset)
        print(f'Test accuracy: {test_accuracy:.4f}')
        ```

2. **Visualize Training History**
    - **TensorFlow**
        ```python
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.legend(loc='lower right')
        plt.show()
        ```
    - **PyTorch**
        To visualize training history in PyTorch, you can log the metrics manually during training and plot them. Here's an example:
        ```python
        # Example of logging
        train_losses = []
        test_losses = []
        test_accuracies = []

        for epoch in range(num_epochs):
            # Training loop
            model.train()
            running_loss = 0.0
            for data, target in train_loader:
                # Training steps...
                running_loss += loss.item()
            train_losses.append(running_loss / len(train_loader))

            # Evaluation loop
            model.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    # Evaluation steps...
                    test_loss += criterion(output, target).item()
                    correct += pred.eq(target.view_as(pred)).sum().item()
            test_losses.append(test_loss / len(test_loader.dataset))
            test_accuracies.append(correct / len(test_loader.dataset))

        # Plotting
        epochs = range(1, num_epochs + 1)
        plt.plot(epochs, train_losses, label='Training Loss')
        plt.plot(epochs, test_losses, label='Test Loss')
        plt.plot(epochs, test_accuracies, label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Metrics')
        plt.legend()
        plt.show()
        ```

## Project Structure
