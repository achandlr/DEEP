import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from abc import ABC, abstractmethod
import time
import logging
from src.models.ModelBaseClass import ModelBaseClass
from itertools import product
import random

def prepare_data(X_train, X_test, Y_train, batch_size=64):
    """
    Prepare the data for training and testing.

    Args:
        X_train (numpy.ndarray): Training input data.
        X_test (numpy.ndarray): Testing input data.
        Y_train (numpy.ndarray): Training target data.
        batch_size (int, optional): Batch size for data loaders. Defaults to 64.

    Returns:
        tuple: A tuple containing the training data loader and testing data loader.
    """
    X_train_tensor = torch.tensor(X_train.astype(np.float32)) if X_train is not None else None
    Y_train_tensor = torch.tensor(Y_train.astype(np.float32)) if Y_train is not None else None
    X_test_tensor = torch.tensor(X_test.astype(np.float32)) if X_test is not None else None

    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor) if X_train_tensor is not None and Y_train_tensor is not None else None
    test_dataset = TensorDataset(X_test_tensor) if X_test_tensor is not None else None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) if train_dataset is not None else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) if test_dataset is not None else None

    return train_loader, test_loader

def train_model(model, train_loader, epochs=100, learning_rate=0.001, l2_regularization=None, optimizer_type='Adam', show_loss=False):
    """
    Train the model.

    Args:
        model (nn.Module): The PyTorch model to train.
        train_loader (DataLoader): The data loader for training data.
        epochs (int, optional): Number of training epochs. Defaults to 100.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.001.
        l2_regularization (float, optional): L2 regularization strength. Defaults to None.
        optimizer_type (str, optional): Type of optimizer to use. Defaults to 'Adam'.
        show_loss (bool, optional): Whether to print the loss during training. Defaults to False.
    """
    if optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_regularization if l2_regularization is not None else 0)
    elif optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=l2_regularization if l2_regularization is not None else 0)
    else:
        raise ValueError("Unsupported optimizer type. Choose 'Adam' or 'SGD'.")

    criterion = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.squeeze(), target)
            loss.backward()
            optimizer.step()
        if show_loss:
            print(f"Epoch {epoch+1}, Loss: {loss.item()}")

def test_model(model, test_loader):
    """
    Test the model.

    Args:
        model (nn.Module): The trained PyTorch model.
        test_loader (DataLoader): The data loader for testing data.

    Returns:
        numpy.ndarray: Predicted labels for the test data.
    """
    model.eval()
    y_pred = []

    with torch.no_grad():
        for data in test_loader:
            output = model(data[0])
            predicted = (output.squeeze() > 0.5).float()
            predicted_list = predicted.tolist()
            if isinstance(predicted_list, float):
                predicted_list = [predicted_list]
            y_pred.extend(predicted_list)

    y_pred = np.array(y_pred)

    return y_pred

class NeuralModel2Layer(ModelBaseClass, nn.Module):
    """
    A 2-layer neural network model.

    Args:
        intermediate_size (int, optional): Size of the intermediate layer. Defaults to 50.
        batch_size (int, optional): Batch size for data loaders. Defaults to 64.
        epochs (int, optional): Number of training epochs. Defaults to 100.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.001.
        dropout_rate (float, optional): Dropout rate for the intermediate layer. Defaults to None.
        l2_regularization (float, optional): L2 regularization strength. Defaults to None.
        use_batch_norm (bool, optional): Whether to use batch normalization. Defaults to False.
        optimizer_type (str, optional): Type of optimizer to use. Defaults to 'Adam'.
    """
    def __init__(self, intermediate_size=50, batch_size=64, epochs=100, learning_rate=0.001, dropout_rate=None, l2_regularization=None, use_batch_norm=False, optimizer_type='Adam'):
        super(NeuralModel2Layer, self).__init__()
        self.intermediate_size = intermediate_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.l2_regularization = l2_regularization
        self.use_batch_norm = use_batch_norm
        self.optimizer_type = optimizer_type

    def initialize_layers(self, input_size):
        self.fc1 = nn.Linear(input_size, self.intermediate_size)
        self.fc2 = nn.Linear(self.intermediate_size, 1)
        self.dropout = nn.Dropout(self.dropout_rate) if self.dropout_rate is not None else None
        self.batch_norm = nn.BatchNorm1d(self.intermediate_size) if self.use_batch_norm else None

    def forward(self, x):
        x = self.fc1(x)
        if self.use_batch_norm:
            x = self.batch_norm(x)
        x = F.relu(x)
        if self.dropout:
            x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

    def _train(self, X_train, Y_train):
        input_size = X_train.shape[1]
        self.initialize_layers(input_size)

        train_loader, _ = prepare_data(X_train, None, Y_train, batch_size=self.batch_size)
        train_model(self, train_loader, epochs=self.epochs, learning_rate=self.learning_rate, l2_regularization=self.l2_regularization, optimizer_type=self.optimizer_type)

    def predict(self, X):
        _, test_loader = prepare_data(None, X, None, batch_size=self.batch_size)
        return test_model(self, test_loader)

    def report_trained_parameters(self):
        trained_paramaters = ""
        for name, param in self.named_parameters():
            if param.requires_grad:
                trained_paramaters += f"{name}: {param.data}\n"
        return trained_paramaters

def load_pytorch_models():
    """
    Load a list of PyTorch models with different parameter variations.

    Returns:
        list: A list of PyTorch models.
    """
    pytorch_models = []
    baseline_params = {
        'intermediate_size': 32, 'batch_size': 16, 'epochs': 20,
        'learning_rate': 0.001, 'dropout_rate': 0.2, 'l2_regularization': 0.01,
        'use_batch_norm': True, 'optimizer_type': 'Adam'
    }

    groups = {
        'intermediate_size': [8, 16, 32, 64],
        'batch_size': [4, 8, 16, 32],
        'epochs': [5, 20, 100],
        'learning_rate': [0.001, 0.01],
        'dropout_rate': [None, 0.1, 0.2, 0.4],
        'l2_regularization': [0.01, 0.001],
        'use_batch_norm': [True, False],
        'optimizer_type': ['Adam', 'SGD']
    }

    for param, values in groups.items():
        for value in values:
            new_params = baseline_params.copy()
            new_params[param] = value
            model = NeuralModel2Layer(**new_params)
            pytorch_models.append(model)

    assert len(pytorch_models) == sum(len(values) for values in groups.values())

    return pytorch_models
