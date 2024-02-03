#importing libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score

#Feature Engineering
#Select feature and target
#checking dimension 
print("Number of dimensions:", X.ndim)
print("Number of dimensions:", y.ndim)

#Standardisation
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

#function for he initialisation
def initialise_parameters_he(X_neurons, hiddenLayer_neuron_1, hiddenLayer_neuron_2, y_neurons):
    #reproducible results
    np.random.seed(2)  
    parameters = {
        #input layer
        #weight and bias for layer one 
        "w1": np.random.randn(hiddenLayer_neuron_1, X_neurons) * np.sqrt(2. / X_neurons),
        "b1": np.zeros((hiddenLayer_neuron_1, 1)),
        #hidden layer
        #weight and bias for layer two
        "w2": np.random.randn(hiddenLayer_neuron_2, hiddenLayer_neuron_1) * np.sqrt(2. / hiddenLayer_neuron_1),
        "b2": np.zeros((hiddenLayer_neuron_2, 1)),
        #output layer
        #weight and bias for layer three
        "w3": np.random.randn(y_neurons, hiddenLayer_neuron_2) * np.sqrt(2. / hiddenLayer_neuron_2),
        "b3": np.zeros((y_neurons, 1))
    }
    #returning parameters
    return parameters

#input shape, two rows for features
X_neurons = X.shape[1]
print(f"Input shape: {X_neurons}")

#calling function to check parameters
parameters = initialise_parameters_he(X_neurons, 2, 2, 1)
print(parameters)

#activation functions
#sigmoid 
def sigmoid(z):
    return 1/ (1 +  np.exp(-z))
#relu 
def relu(z):
    return np.maximum(0, z)

#forward propagation 
def forward_propagation(X, parameters):
    #transpose feature matrix to match the shape expected by the neural network
    X = X.T
    #declaring local variable for weights and bias
    #layer 1, layer 2 and layer 3 parameters (weight and bias)
    weight_1, bias_1 = parameters['w1'], parameters['b1']
    weight_2, bias_2 = parameters['w2'], parameters['b2']
    weight_3, bias_3 = parameters['w3'], parameters['b3']
    
    #pre-activation or Z1 value for the first layer using the dot product of weights and input plus the bias
    #applying the ReLU activation function to the pre-activation values for layer 1
    pre_activation_layer_1 = np.dot(weight_1, X) + bias_1
    activation_function_layer_1 = relu(pre_activation_layer_1)
    #pre-activation or Z2 value for the second layer using the dot product of weights and input plus the bias
    #applying the ReLU activation function to the pre-activation values for layer 2
    pre_activation_layer_2 = np.dot(weight_2, activation_function_layer_1) + bias_2
    activation_function_layer_2 = relu(pre_activation_layer_2)
    #pre-activation or Z3 value for the second layer using the dot product of weights and input plus the bias
    #applying the sigmoid activation function to the pre-activation values for layer 3
    pre_activation_layer_3 = np.dot(weight_3, activation_function_layer_2) + bias_3
    activation_function_layer_3 = sigmoid(pre_activation_layer_3)  
    
    #cache data of pre-activation and activation values of each layer for use in backpropagation
    cache = { "z1": pre_activation_layer_1, "a1": activation_function_layer_1,
              "z2":pre_activation_layer_2,  "a2": activation_function_layer_2,
              "z3":pre_activation_layer_3,  "a3": activation_function_layer_3
            }
    return activation_function_layer_3, cache


#performing forward propagataion 
activation_function_layer_3, cache = forward_propagation(X, parameters)
activation_function_layer_3

#declaring function computing loss 
def computing_loss(activation_function_layer_3, y):
    #activation_function_layer_3 is shaped (1, m) to match y when y is reshaped to (1, m)
    #if one-dimensional and reshaped two-dimensional y
    #small value to avoid log(0)
    m = y.size  
    epsilon = 1e-5  
    #ensure Y is reshaped to (1, m) for compatibility with activation_function_layer_3
    if y.ndim == 1:
        y = y.reshape(1, m)
    #adjusting the computation to avoid log(0)
    log_probs = np.log(activation_function_layer_3 + epsilon) * y + np.log(1 - activation_function_layer_3 + epsilon) * (1 - y)
    cost = -np.sum(log_probs) / m
    #ensuring cost is the proper shape regardless of its original form
    cost = np.squeeze(cost)  
    return cost

#performing computational loss
computing_loss(activation_function_layer_3, y)

#function for backward propagation
def backward_propagation(parameters, cache, X, y):
    #total number of sample
    #X.shape[0] <- first dimension
    #X.shape[1] <- second dimension
    m = X.shape[1]
    
    #retrieve weights and activations
    weight_1, weight_2, weight_3 = parameters["w1"], parameters["w2"], parameters["w3"]
    activation_1, activation_2, activation_3 = cache["a1"], cache["a2"], cache["a3"]
    
    #backward propagation for layer 3
    derivative_pre_activation_layer_3 = activation_3 - y
    derivative_weight_layer_3 = np.dot(derivative_pre_activation_layer_3, activation_2.T) / m
    derivative_bias_layer_3 = np.sum(derivative_pre_activation_layer_3, axis=1, keepdims=True) / m
    
    #backward propagation for layer 2
    #include the derivative of the activation function used in layer 2
    #ReLU was used in layer 2
    derivative_activation_2 = np.dot(weight_3.T, derivative_pre_activation_layer_3)
    #derivative of ReLU
    derivative_pre_activation_layer_2 = derivative_activation_2 * (activation_2 > 0)  
    derivative_weight_layer_2 = np.dot(derivative_pre_activation_layer_2, activation_1.T) / m
    derivative_bias_layer_2 = np.sum(derivative_pre_activation_layer_2, axis=1, keepdims=True) / m
    
    #backward propagation for layer 1
    #assuming ReLU was used in layer 1
    derivative_activation_1 = np.dot(weight_2.T, derivative_pre_activation_layer_2)
    #derivative of ReLU
    derivative_pre_activation_layer_1 = derivative_activation_1 * (activation_1 > 0)  
    derivative_weight_layer_1 = np.dot(derivative_pre_activation_layer_1, X) / m
    derivative_bias_layer_1 = np.sum(derivative_pre_activation_layer_1, axis=1, keepdims=True) / m

    gradients = {
        "dw1": derivative_weight_layer_1,
        "db1": derivative_bias_layer_1,
        "dw2": derivative_weight_layer_2,
        "db2": derivative_bias_layer_2,
        "dw3": derivative_weight_layer_3,
        "db3": derivative_bias_layer_3
    }
    return gradients

#calcualting gradients using backward propagation 
grads = backward_propagation(parameters, cache, X, y)
print(grads)

#declaring function for updating parameters
def update_parameters(parameters, grads, learning_rate=0.001):
    #number of layers
    num_layers = len(parameters) // 2  
    #for loop to iterate over range of layer + 1
    for i in range(1, num_layers + 1):
        #key names to match those used in parameters and grads dictionaries
        parameters["w" + str(i)] -= learning_rate * grads["dw" + str(i)]
        parameters["b" + str(i)] -= learning_rate * grads["db" + str(i)]
    return parameters

#calculating update parameters using update parameter function
print(update_parameters(parameters, grads, learning_rate=0.01))

#function to train model
def train_model(X, y, iterations=1000, learning_rate=0.001):
    #reproducibility
    np.random.seed(1)  
    #He initialization
    parameters = initialise_parameters_he(X.shape[1], 5, 3, 1)
    #loss and accuracy values 
    loss_values = []  
    accuracy_values = []  
    for i in range(iterations):
        #forward propagation
        activation_layers, caches = forward_propagation(X, parameters)
        #compute and store loss
        cost = computing_loss(activation_layers, y)
        loss_values.append(cost)
        #backward propagation
        grads = backward_propagation(parameters, caches, X, y)
        #update parameters
        parameters = update_parameters(parameters, grads, learning_rate)
        
        #predictions and accuracy 
        predictions = (activation_layers > 0.5).astype(int)
         #ensuring shapes match
        accuracy = np.mean(predictions == y.reshape(predictions.shape)) * 100 
        accuracy_values.append(accuracy)
    return parameters, loss_values, accuracy_values

#splitting data into trianing and testing
#80 percent train and 20 percent test
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

#checking shape of train set
(X_train.shape, y_train.shape)
#checking shape of test set
(X_test.shape, y_test.shape)

#calling function to calcualte parameters, loss and accuracy values
parameters, loss_values, accuracy_values = train_model(X_train, y_train, iterations=500, learning_rate=0.001)

#loss plot
plt.figure(figsize=(10, 3))
plt.subplot(1, 2, 1)
plt.plot(loss_values, lw=2,  label='Loss')
plt.title('Loss over iterations')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.grid(True, ls='--', alpha=0.5)
plt.legend()
#accu
plt.subplot(1, 2, 2)
plt.plot(accuracy_values, label='Accuracy', color='orange', lw=2)
plt.title('Accuracy over iterations')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.grid(True, ls='--', alpha=0.5)
plt.legend()

plt.tight_layout()
plt.show()

def model_predict(X, parameters):
    #forward propagation to predict probabilities
    activation_layer, i = forward_propagation(X, parameters)
    #convert probabilities to 0 or 1 using 0.5 as a threshold
    predictions = activation_layer > 0.5
    return predictions
def compute_accuracy(predictions, y):
    #ensure y is reshaped for comparison
    y = y.reshape(predictions.shape)
    #number of correct predictions
    correct_predictions = np.sum(predictions == y)
    #accuracy by dividing correct predictions by total number of predictions
    #y.shape[0] if y is shaped as (m, 1)
    accuracy = correct_predictions / y.shape[1]  
    #accuracy in percentage
    return accuracy * 100 
  
#making predictions 
predictions = model_predict(X_test, parameters)
predictions[0][:20]

#calculating accuracy
accuracy = compute_accuracy(predictions, y_test)
print(f"Accuracy: {accuracy}%")

def plot_decision_boundary(model, X, y):
    #min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    #size in the mesh
    h = 0.01  
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    #Flatten the grid to pass into the model
    Z = model(np.c_[xx.ravel(), yy.ravel()].T, parameters)
    #Threshold to get binary predictions
    Z = Z > 0.5  
    Z = Z.reshape(xx.shape)
    #Plot the contour
    plt.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.Spectral)
    #markers for Class 0 and Class 1
    class0_color = 'blue'
    class0_marker = 'o'
    class1_color = 'red'
    class1_marker = 's'
    #Class 0 and Class 1
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c=class0_color, marker=class0_marker, label='Class 0')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c=class1_color, marker=class1_marker, label='Class 1')
    #decision boundary
    plt.contour(xx, yy, Z, colors='k', linewidths=0.5)
    plt.xlabel('Age')
    plt.ylabel('Annual Salary')
    plt.title("Decision Boundary")
    custom_legend_labels = {
        'Class 0': {'color': class0_color, 'marker': class0_marker},
        'Class 1': {'color': class1_color, 'marker': class1_marker}
    }
    handles = [plt.Line2D([], [], marker=properties['marker'], color=properties['color'], 
                          label=label, markersize=10) 
               for label, properties in custom_legend_labels.items()]
    
    plt.legend(handles=handles, loc='best')
    plt.grid(True, ls='--', alpha=0.2, color='black')
    plt.show()

    
#predict function for the decision boundary plotting
def model_predict(X, parameters):
    X = X.T
    #run forward propagation to predict probabilities
    activation_layer, i = forward_propagation(X, parameters)
    #converting probabilities to 0 or 1 using 0.5 as a threshold
    predictions = activation_layer > 0.5
    return predictions
  
#decision boundary for the test set
#X_test.T to match the expected input shape
plot_decision_boundary(model_predict, X_test, y_test)  

#reshaping predictions
predictions_reshaped = predictions.reshape(-1).astype(int)
#calcualting confusion matrix
conf_matrix = confusion_matrix(y_test, predictions_reshaped)

#manually calcaulting accuracy, precision, recall and f1 score
tn, fp, fn, tp = conf_matrix.ravel()
true_predictions = tp + tn
false_predictions = fp + fn
total_predictions = true_predictions + false_predictions
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * (precision * recall) / (precision + recall)

#printing result
print(f"True Positives (TP): {tp}")
print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")

#total predictions
print(f"Total True Predictions (TP + TN): {true_predictions}")
print(f"Total False Predictions (FP + FN): {false_predictions}")
print(f"Total Predictions: {total_predictions}")

#classification metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score}")

def model_predict_proba(X, parameters):
    #forward propagation to predict probabilities
    activation_layer, _ = forward_propagation(X, parameters)
    #return the probabilities without applying a threshold
    return activation_layer
  
y_proba = model_predict_proba(X_test, parameters).reshape(-1)
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='darkgray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.grid(True, ls='--', alpha=0.2, color='black')
plt.legend(loc='lower right')
plt.show()

# ******************************************************
# Author: Hemant Thapa
# Programming Language: Python
# Date Pushed to GitHub: 03.02.2024
# Email: hemantthapa1998@gmail.com
# ******************************************************
