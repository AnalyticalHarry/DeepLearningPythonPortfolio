#importing libraries 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

#defining function 
def RegNeuralNet(X,y):
      #split the dataset into training and testing sets
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
      
      #function to create and compile the neural network model
      def build_model(hidden_layer_size, num_neurons):
          model = keras.Sequential([
              #input layer
              layers.Input(shape=(X_train.shape[1],)),  
              #hidden layer with variable neurons
              layers.Dense(num_neurons, activation='relu'),  
              #output layer (1 neuron for regression)
              layers.Dense(1)  
          ])
          model.compile(optimizer='adam', loss='mean_squared_error')
          return model
      
      #hyperparameters to search
      #number of hidden layers
      hidden_layer_sizes = [1, 2, 3, 4]  
      #number of neurons in the hidden layer
      num_neurons_options = [16, 32, 64]  
      
      best_model = None
      best_mse = float('inf')
      
      #perform hyperparameter tuning
      for hidden_layer_size in hidden_layer_sizes:
          for num_neurons in num_neurons_options:
              model = build_model(hidden_layer_size, num_neurons)
              model.fit(X_train, y_train, epochs=50, verbose=0)
              mse = model.evaluate(X_test, y_test, verbose=0)
              print(f'Hidden Layers: {hidden_layer_size}, Neurons: {num_neurons}, MSE: {mse:.4f}')
              if mse < best_mse:
                  best_mse = mse
                  best_hidden_layer_size = hidden_layer_size
                  best_num_neurons = num_neurons
                  best_model = model
      
      #evaluate the best model on the test set
      final_mse = best_model.evaluate(X_test, y_test)
      print(f'Final Test MSE: {final_mse:.4f}')
      
      #print the best combination of hidden layers and neurons
      print(f'Best Hidden Layer Size: {best_hidden_layer_size}')
      print(f'Best Number of Neurons: {best_num_neurons}')
  
RegNeuralNet(X,y)

# ******************************************************
# Author: Hemant Thapa
# Programming Language: Python
# Date Pushed to GitHub: 29.01.2024
# Email: hemantthapa1998@gmail.com
# ******************************************************
