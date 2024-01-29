#importing libraries 
import itertools
import random 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


def hyperparameter_tuning(X, y):
    #splitting the dataset into training and testing sets
    def train_test_split_np(X, y, test_size=0.2, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)
        num_samples = X.shape[0]
        num_test_samples = int(test_size * num_samples)
        #shuffle the data indices
        shuffled_indices = np.random.permutation(num_samples)
        #split the data into training and testing sets
        X_train = X[shuffled_indices[num_test_samples:]]
        y_train = y[shuffled_indices[num_test_samples:]]
        X_test = X[shuffled_indices[:num_test_samples]]
        y_test = y[shuffled_indices[:num_test_samples]]
        return X_train, X_test, y_train, y_test
    
    X_train, X_test, y_train, y_test = train_test_split_np(X, y, test_size=0.2, random_state=42)

    #create and compile the neural network model
    def build_model(hidden_layer_size, num_neurons, activations, optimizer):
        model = keras.Sequential()
        #input layer
        model.add(layers.Input(shape=(X_train.shape[1],)))  
        for i in range(hidden_layer_size):
            #activation functions for each hidden layer
            model.add(layers.Dense(num_neurons, activation=activations[i]))
        #output layer
        model.add(layers.Dense(1)) 
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        return model

    #define hyperparameters to search
    #number of hidden layers
    hidden_layer_sizes = [x for x in range(1, 6)]
    #number of neurons 
    num_neurons_options = [16, 32, 64]
    #activation functions 
    activation_functions = ['relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu', 'swish', 'selu', 'gelu', 'softmax']
    #optimizers
    optimizers = ['adam', 'sgd', 'rmsprop', 'adagrad', 'nadam', 'adadelta']

    #declaring initial parameters
    best_model = None
    best_mse = float('inf')
    best_hidden_layer_size = None
    best_num_neurons = None
    best_activations = None
    best_optimizer = None

    #perform hyperparameter tuning
    for hidden_layer_size in hidden_layer_sizes:
        for num_neurons in num_neurons_options:
            #generate all possible activation combinations for the given hidden layer size
            for activations in itertools.product(activation_functions, repeat=hidden_layer_size):
                for optimizer in optimizers:
                    model = build_model(hidden_layer_size, num_neurons, activations, optimizer)
                    history = model.fit(X_train, y_train, validation_split=0.2, epochs=1, verbose=0)
                    mse = model.evaluate(X_test, y_test, verbose=0)
                    print(f'Hidden Layers: {hidden_layer_size}, Neurons: {num_neurons}, Activations: {activations}, Optimizer: {optimizer}, MSE: {mse:.4f}')
                    if mse < best_mse:
                        best_mse = mse
                        best_hidden_layer_size = hidden_layer_size
                        best_num_neurons = num_neurons
                        best_activations = activations
                        best_optimizer = optimizer
                        best_model = model
                        best_training_loss = history.history['loss']
                        best_validation_loss = history.history['val_loss']

    #evaluate the best model on the test set
    final_mse = best_model.evaluate(X_test, y_test)
    print(f'Final Test MSE: {final_mse:.4f}')
    #best combination of hidden layers, neurons, activations, and optimizers
    print(f'Best Hidden Layer Size: {best_hidden_layer_size}')
    print(f'Best Number of Neurons: {best_num_neurons}')
    print(f'Best Activations: {best_activations}')
    print(f'Best Optimizer: {best_optimizer}')

    #plot the training and validation loss for the best model
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(best_training_loss) + 1), best_training_loss, label='Best Model Training Loss', linestyle='--')
    plt.plot(range(1, len(best_validation_loss) + 1), best_validation_loss, label='Best Model Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss for Best Model')
    plt.legend()
    plt.grid(True)
    plt.show()

hyperparameter_tuning(X, y)
# ******************************************************
# Author: Hemant Thapa
# Programming Language: Python
# Date Pushed to GitHub: 29.01.2024
# Email: hemantthapa1998@gmail.com
# ******************************************************
