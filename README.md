# Multilayer Perceptron
My own take on a simple multilayer perceptron neural network.

## Files
- `data.csv` contains dummy data.
- `Neuron.py`, `Layer.py`, `NeuralNetwork.py` contain the logic for their respective components.
- `Predict.py` contains an example on how to use the network to predict new values, after it has been trained.
- `Train.py` contains an example on how to train the network.

## Documentation
**Creating the neural network...**
```python
nn = NeuralNetwork(inputs, layers, outputs)
```
where:
- `inputs` is an integer indicating the number nodes in the input layer (in other words, the number of features).
- `layers` is an array of integers indicating the number of nodes in each hidden layer.
	- Example: [5,10] would build a network with 2 hidden layers. First hidden layer has 5 nodes; second one has 10.
- `outputs` is an integer indicating the number of outputs for the network.

(*Note:* The input layer and all hidden layers automatically get a bias node. It does not need to be specified in the parameter).

**Changing network parameters...**
```python
nn.lam = 0.7
nn.learning_rate = 0.6
nn.momentum_rate = 0.1
```
After instantiating the network, there are three parameters that can be changed to alter the training process.
- `lam` specifies the lambda value used for the sigmoid activation function.
	- f(x) = 1 / (1 + exp(-x · lam)).
- `learning_rate` specifies the learning rate for weight updates after an epoch.
- `momentum_rate` specifies the momentum rate for weight updates after an epoch.

**Training the neural network...**
```python
train_error, val_error = nn.train(x, y, val_split, epochs, min_delta, patience, restore_best_weights)
```
where, for parameters:
- `x` is the input data as a matrix. Each row corresponds to a data sample; each column corresponds to a given feature. 
	- *The number of columns should be equal to the* `inputs` *argument used when instantiating the network.*
- `y` is the target values as a matrix. Each row corresponds to a row from the input data (paired sequentially); each column corresponds to an output for a given row. 
	- *The number of rows should be equal to the number of rows on the x argument.*
	- *The number of columns should be equal to the* `outputs` *argument used when instantiating the network.*
- `val_split` is the percentage of data used to perform cross-validation, expressed as a decimal.
- `epochs` is the maximum number of epochs used to train the data.
- `min_delta` is the minimum amount of change that is still considered an improvement during validation training.
- `patience` is the number of epochs with no improvement needed to stop training. Gets reset after an epoch of improvement.
- `restore_best_weights` is a boolean that indicates whether the model should keep the weights that achieved the lowest validation error.

and, for the return tuple:
- `train_error` is an array containing the training error (RMSE) for each of the epochs.
- `val_error`  is an array containing the validation error (RMSE) for each of the epochs.

**Extracting the neural network weights...**
```python
nn.weights
```
After training, this line will contain the final weights of the neural network (expressed as a 3-dimensional tuple; more on next section).

**Setting the neural network weights...**
```python
nn = NeuralNetwork(2, [4, 4], 2)
nn.setWeights(weights)
```
If the network is instantiated again or in a new module, the weights can be loaded by using `setWeights()` method, where:
- `weights`  is a 3-dimensional tensor containing the weights of the neural network. First dimension represents the connections between two adjacent layers (e.g. weights from Input Layer to Hidden Layer #1). Second dimension represents each of the nodes in the giving-end of a weight (e.g. Bias Node, Input Node #1, Input Node #2, etc.). First element in this dimension always represents the bias for a given layer. Third dimension represents each of the nodes in the receiving-end of a weight (e.g. Hidden Node #1 for Layer #1, Hidden Node #2 for Layer #1, etc.). Notice, that there is no Bias Node in the third dimension. Refer to  `Predict.py` for a labeled example.

**Making a prediction with the neural network ...**
```python
nn.feedForward(inputs)
```
where:
- `inputs` is an array containing all the features for a given sample. 
	- *Its length should match that of  the input argument used when instantiating the network.*
