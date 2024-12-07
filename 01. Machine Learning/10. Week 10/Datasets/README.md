# Introduction to MLP with TensorFlow and PyTorch

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.x-brightgreen.svg)
![PyTorch](https://img.shields.io/badge/pytorch-1.x-brightgreen.svg)

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Datasets](#datasets)
  - [Heart Disease](#heart-disease)
  - [Iris](#iris)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
- [Usage](#usage)
  - [Running on Google Colab](#running-on-google-colab)
  - [Training the MLP with TensorFlow](#training-the-mlp-with-tensorflow)
    - [For Heart Disease](#for-heart-disease-tensorflow)
    - [For Iris](#for-iris-tensorflow)
  - [Training the MLP with PyTorch](#training-the-mlp-with-pytorch)
    - [For Heart Disease](#for-heart-disease-pytorch)
    - [For Iris](#for-iris-pytorch)
  - [Evaluating the Model](#evaluating-the-model)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction
Welcome to the **Introduction to Multi-Layer Perceptron (MLP) with TensorFlow and PyTorch** repository! This project provides comprehensive guides to building, training, and evaluating MLP models using the **Heart Disease** and **Iris** datasets. Leveraging the power of **TensorFlow** and **PyTorch** alongside Google Colab's free computational resources, you can efficiently develop and experiment with neural network models without requiring extensive local computational power.

## Features
- **Comprehensive Jupyter Notebooks**: Detailed notebooks guiding you through the MLP implementation in both TensorFlow and PyTorch.
- **Support for Heart Disease and Iris Datasets**: Two diverse datasets for classification tasks.
- **Google Colab Integration**: Easy setup and execution on Colab with GPU acceleration.
- **Performance Visualizations**: Graphs to visualize training metrics and model performance.
- **Modular and Extensible Code**: Easily adaptable for other datasets and neural network architectures.

## Datasets

### Heart Disease
- **Description**: The Heart Disease dataset contains various medical attributes of patients and indicates the presence or absence of heart disease. It is commonly used for binary classification tasks.
- **Number of Instances**: 303
- **Number of Features**: 13
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+Disease)
- **Use Case**: Predicting the likelihood of heart disease based on patient attributes.

### Iris
- **Description**: The Iris dataset consists of 150 samples of iris flowers from three different species. Each sample has four features representing the dimensions of the flowers.
- **Number of Instances**: 150
- **Number of Features**: 4
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris)
- **Use Case**: Multiclass classification of iris species based on flower measurements.

## Getting Started

### Prerequisites
- **Google Account**: To access Google Colab.
- **Web Browser**: Latest version of Chrome, Firefox, or Safari.

### Setup
1. **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/mlp-tensorflow-pytorch-colab.git
    ```
2. **Navigate to the Project Directory**
    ```bash
    cd mlp-tensorflow-pytorch-colab
    ```
3. **Open the Notebook in Google Colab**
    - Go to [Google Colab](https://colab.research.google.com/).
    - Click on `File` > `Upload notebook`.
    - Upload the desired notebook:
      - `tensorflow/heart_disease_mlp_tf.ipynb` or `tensorflow/iris_mlp_tf.ipynb` for TensorFlow.
      - `pytorch/heart_disease_mlp_pt.ipynb` or `pytorch/iris_mlp_pt.ipynb` for PyTorch.

## Usage

### Running on Google Colab
1. **Ensure GPU is Enabled**
    - In Colab, go to `Runtime` > `Change runtime type`.
    - Select `GPU` from the Hardware accelerator dropdown.
    - Click `Save`.

2. **Install Required Libraries**
    - **TensorFlow**
        ```python
        !pip install tensorflow matplotlib pandas scikit-learn
        ```
    - **PyTorch**
        ```python
        !pip install torch torchvision matplotlib pandas scikit-learn
        ```

3. **Import Libraries**
    - **TensorFlow**
        ```python
        import tensorflow as tf
        from tensorflow.keras import layers, models
        import matplotlib.pyplot as plt
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        ```
    - **PyTorch**
        ```python
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torch.optim as optim
        from torch.utils.data import Dataset, DataLoader
        import matplotlib.pyplot as plt
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        ```

### Training the MLP with TensorFlow

#### For Heart Disease (TensorFlow)
1. **Load the Dataset**
    ```python
    # Load dataset
    url = 'https://raw.githubusercontent.com/your-username/datasets/main/heart.csv'
    data = pd.read_csv(url)

    # Features and Labels
    X = data.drop('target', axis=1).values
    y = data['target'].values

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    ```

2. **Build the MLP Model**
    ```python
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    ```

3. **Compile the Model**
    ```python
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    ```

4. **Train the Model**
    ```python
    history = model.fit(X_train, y_train, epochs=50, 
                        batch_size=32, 
                        validation_data=(X_test, y_test))
    ```

#### For Iris (TensorFlow)
1. **Load the Dataset**
    ```python
    from tensorflow.keras.utils import to_categorical

    # Load dataset
    url = 'https://raw.githubusercontent.com/your-username/datasets/main/iris.csv'
    data = pd.read_csv(url)

    # Features and Labels
    X = data.drop('species', axis=1).values
    y = pd.get_dummies(data['species']).values

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    ```

2. **Build the MLP Model**
    ```python
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])
    ```

3. **Compile the Model**
    ```python
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    ```

4. **Train the Model**
    ```python
    history = model.fit(X_train, y_train, epochs=100, 
                        batch_size=16, 
                        validation_data=(X_test, y_test))
    ```

### Training the MLP with PyTorch

#### For Heart Disease (PyTorch)
1. **Load the Dataset**
    ```python
    class HeartDiseaseDataset(Dataset):
        def __init__(self, csv_file):
            self.data = pd.read_csv(csv_file)
            self.X = self.data.drop('target', axis=1).values
            self.y = self.data['target'].values
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(self.X)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            sample = torch.tensor(self.X[idx], dtype=torch.float32)
            label = torch.tensor(self.y[idx], dtype=torch.float32)
            return sample, label

    # Load datasets
    train_dataset = HeartDiseaseDataset('https://raw.githubusercontent.com/your-username/datasets/main/heart.csv')
    train_size = int(0.8 * len(train_dataset))
    test_size = len(train_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    ```

2. **Define the Model**
    ```python
    class MLP(nn.Module):
        def __init__(self, input_size):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(input_size, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 1)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = torch.sigmoid(self.fc3(x))
            return x

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_size=train_dataset.dataset.X.shape[1]).to(device)
    ```

3. **Define Loss and Optimizer**
    ```python
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    ```

4. **Train the Model**
    ```python
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    ```

#### For Iris (PyTorch)
1. **Load the Dataset**
    ```python
    class IrisDataset(Dataset):
        def __init__(self, csv_file):
            self.data = pd.read_csv(csv_file)
            self.X = self.data.drop('species', axis=1).values
            self.y = pd.get_dummies(self.data['species']).values
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(self.X)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            sample = torch.tensor(self.X[idx], dtype=torch.float32)
            label = torch.tensor(self.y[idx], dtype=torch.float32)
            return sample, label

    # Load datasets
    iris_dataset = IrisDataset('https://raw.githubusercontent.com/your-username/datasets/main/iris.csv')
    train_size = int(0.8 * len(iris_dataset))
    test_size = len(iris_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(iris_dataset, [train_size, test_size])

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    ```

2. **Define the Model**
    ```python
    class MLP(nn.Module):
        def __init__(self, input_size, num_classes):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(input_size, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, num_classes)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_size=iris_dataset.dataset.X.shape[1], num_classes=3).to(device)
    ```

3. **Define Loss and Optimizer**
    ```python
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    ```

4. **Train the Model**
    ```python
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), torch.argmax(target, dim=1).to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    ```

### Evaluating the Model
1. **Evaluate on Test Data**
    - **TensorFlow**
        ```python
        # For Heart Disease
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
        print(f'Test Accuracy: {test_acc:.4f}')

        # For Iris
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
        print(f'Test Accuracy: {test_acc:.4f}')
        ```
    - **PyTorch**
        ```python
        # For Heart Disease
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device).unsqueeze(1)
                outputs = model(data)
                loss = criterion(outputs, target)
                test_loss += loss.item()
                preds = (outputs >= 0.5).float()
                correct += (preds == target).sum().item()
        
        test_loss /= len(test_loader)
        test_accuracy = correct / len(test_loader.dataset)
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

        # For Iris
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                test_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == target).sum().item()
        
        test_loss /= len(test_loader)
        test_accuracy = correct / len(test_loader.dataset)
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
        ```

2. **Visualize Training History**
    - **TensorFlow**
        ```python
        plt.figure(figsize=(12, 4))

        # Plot Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot Loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()
        ```
    - **PyTorch**
        To visualize training history in PyTorch, you can log the metrics manually during training and plot them. Here's an example:
        ```python
        import matplotlib.pyplot as plt

        train_losses = []
        test_losses = []
        test_accuracies = []

        for epoch in range(num_epochs):
            # Training loop
            model.train()
            running_loss = 0.0
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            train_losses.append(running_loss / len(train_loader))

            # Evaluation loop
            model.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    test_loss += loss.item()
                    preds = (output >= 0.5).float() if isinstance(model, HeartDiseaseModel) else torch.argmax(output, dim=1)
                    correct += (preds == target.view_as(preds)).sum().item()
            test_losses.append(test_loss / len(test_loader))
            test_accuracies.append(correct / len(test_loader.dataset))

            print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {train_losses[-1]:.4f} | Test Loss: {test_losses[-1]:.4f} | Test Accuracy: {test_accuracies[-1]:.4f}')

        # Plotting
        epochs_range = range(1, num_epochs + 1)
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, train_losses, label='Training Loss')
        plt.plot(epochs_range, test_losses, label='Test Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, test_accuracies, label='Test Accuracy')
        plt.title('Test Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()
        ```

