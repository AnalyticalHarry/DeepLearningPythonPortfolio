#import libraies 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import itertools


#split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#create and compile the neural network model for classification
def build_classification_model(hidden_layer_size, num_neurons, activations, optimizer):
    model = keras.Sequential()
    #input layer
    model.add(layers.Input(shape=(X_train.shape[1],)))  
    for i in range(hidden_layer_size):
        #assign different activation functions to each hidden layer
        model.add(layers.Dense(num_neurons, activation=activations[i]))
    #output layer with sigmoid activation for binary classification
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

#hyperparameters to search
hidden_layer_sizes = [1, 2, 3]
num_neurons_options = [16, 32, 64]
activation_functions = ['relu', 'tanh', 'sigmoid']
optimizers = ['adam', 'sgd', 'rmsprop']

best_model = None
best_accuracy = 0.0
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
                model = build_classification_model(hidden_layer_size, num_neurons, activations, optimizer)
                history = model.fit(X_train, y_train, validation_split=0.2, epochs=1, verbose=0)
                _, accuracy = model.evaluate(X_test, y_test, verbose=0)
                print(f'Hidden Layers: {hidden_layer_size}, Neurons: {num_neurons}, Activations: {activations}, Optimizer: {optimizer}, Accuracy: {accuracy:.4f}')
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_hidden_layer_size = hidden_layer_size
                    best_num_neurons = num_neurons
                    best_activations = activations
                    best_optimizer = optimizer
                    best_model = model
                    best_training_accuracy = history.history['accuracy']
                    best_validation_accuracy = history.history['val_accuracy']

#evaluate the best model on the test set
final_accuracy = best_model.evaluate(X_test, y_test, verbose=0)[1]
print(f'Final Test Accuracy: {final_accuracy:.4f}')

#best combination of hidden layers, neurons, activations, and optimizers
print(f'Best Hidden Layer Size: {best_hidden_layer_size}')
print(f'Best Number of Neurons: {best_num_neurons}')
print(f'Best Activations: {best_activations}')
print(f'Best Optimizer: {best_optimizer}')

#plot the training and validation accuracy for the best model
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(best_training_accuracy) + 1), best_training_accuracy, label='Best Model Training Accuracy', linestyle='--')
plt.plot(range(1, len(best_validation_accuracy) + 1), best_validation_accuracy, label='Best Model Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy for Best Model')
plt.legend()
plt.grid(True)
plt.show()

# ******************************************************
# Author: Hemant Thapa
# Programming Language: Python
# Date Pushed to GitHub: 29.01.2024
# Email: hemantthapa1998@gmail.com
# ******************************************************
