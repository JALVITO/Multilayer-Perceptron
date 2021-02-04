import math
import random

from Layer import HiddenLayer, InputLayer, OutputLayer

class NeuralNetwork:
    def __init__(self, inputs, layers, outputs, useLinear = False):
        self.lam = 0.8
        self.learning_rate = 0.8
        self.momentum_rate = 0.1
        self.useLinear = useLinear

        self.layers = []

        # Create output layer
        output_layer = OutputLayer(outputs, useLinear)
        self.layers.insert(0, output_layer) 

        next_layer = self.layers[0]
        layer_num = len(layers)

        # Create hidden layers from back to front
        for layer_size in reversed(layers):
            layer = HiddenLayer(layer_size, next_layer.nonBiasNeurons(), layer_num)
            self.layers.insert(0, layer)

            next_layer = layer
            layer_num -= 1

        # Create input layer
        input_layer = InputLayer(inputs, next_layer.nonBiasNeurons())
        self.layers.insert(0, input_layer)

    @property
    def inputLayer(self):
        return self.layers[0]

    @property
    def outputLayer(self):
        return self.layers[-1]

    @property
    def inputSize(self):
        return self.inputLayer.size() - 1

    @property
    def outputSize(self):
        return self.outputLayer.size()

    @property
    def numberOfLayers(self):
        return len(self.layers)

    @property
    def weights(self):
        return [layers.weights for layers in self.layers[:-1]]

    def cleanLayers(self):
        for layer in self.layers:
            layer.cleanNeurons()

    def setWeights(self, layer_weights):
        if len(layer_weights) != self.numberOfLayers - 1:
            print(f'Number of weight matrices does not correspond. Expected {self.numberOfLayers - 1}, got {len(layer_weights)}.')
            return

        for layer, weights in zip(self.layers[:-1], layer_weights):
            layer.setWeights(weights)

    def feedForward(self, inputs):
        if len(inputs) != self.inputSize:
            print(f'Number of inputs does not correspond. Expected {self.inputSize}, got {len(inputs)}.')
            return

        self.cleanLayers()
        self.inputLayer.setInputs(inputs)

        for layer in self.layers:
            layer.propagateNeurons(self.lam)

        return self.outputLayer.getOutput()

    def backPropagation(self, output, target_output):
        if len(target_output) != self.outputSize:
            print(f'Number of ouputs does not correspond. Expected {self.outputSize}, got {len(target_output)}.')
            return

        # Reverse all but output layers for back propagation
        reversed_layers = list(reversed(self.layers[:-1]))

        for i, layer in enumerate(reversed_layers):
            if i == 0:
                # Local Grad = lambda * activated_output * (1 - activated_output) * error
                grad = [self.lam * o * (1 - o) * (t - o) for o,t in zip(output, target_output)]
            else:
                # Hidden Grad = lambda * activated_val * (1 - activated_val) * sum(local_grad[i] * w[i])
                grad = [self.lam * y * (1 - y) * wg_sum for y, wg_sum in zip(activated_values, weight_grad_summation)]

            # Extract activated values and weights of current layer (before updating) for next layer calculation
            activated_values = [neuron.activationVal for neuron in reversed_layers[i].nonBiasNeurons()]
            layer_weights = [list(neuron.weights.values()) for neuron in reversed_layers[i].nonBiasNeurons()]
            weight_grad_summation = [sum(g*w for g, w in zip(grad, weights)) for weights in layer_weights]

            # Update layer
            layer.updateWeights(grad, self.learning_rate, self.momentum_rate)

    def train(self, x, y, val_split, epochs, min_delta=0, patience=1, restore_best_weights=False):
        train_error = []
        val_error = []

        curr_patience = patience

        min_val_error = math.inf
        best_weights = self.weights

        for epoch in range(epochs):
            train_rmse = 0
            val_rmse = 0

            # Shuffle the patterns
            patterns = list(zip(x, y))
            random.shuffle(patterns)
            x_shuffled, y_shuffled = zip(*patterns)

            # Partition data according to training percentage
            dataset_size = len(x)
            val_cut = int(dataset_size * val_split)

            x_train, y_train = x_shuffled[val_cut:], y_shuffled[val_cut:]
            x_val, y_val = x_shuffled[:val_cut], y_shuffled[:val_cut]

            # Train
            for input_vals, target_vals in zip(x_train, y_train):
                output_vals = self.feedForward(input_vals)
                self.backPropagation(output_vals, target_vals)

            # Calculate train error
            for input_vals, target_vals in zip(x_train, y_train):
                output_vals = self.feedForward(input_vals)
                train_rmse += sum([(t - o) ** 2 for t,o in zip(output_vals, target_vals)])

            epoch_train_error = math.sqrt(train_rmse / len(y_train))
            train_error.append(epoch_train_error)
            
            # Calculate val error
            for input_vals, target_vals in zip(x_val, y_val):
                output_vals = self.feedForward(input_vals)
                val_rmse += sum([(t - o) ** 2 for t,o in zip(output_vals, target_vals)])

            epoch_val_error = math.sqrt(val_rmse / len(y_val))
            val_error.append(epoch_val_error)

            # Store best weights and min val error for restoration
            if restore_best_weights and epoch_val_error < min_val_error:
                min_val_error = epoch_val_error
                best_weights = self.weights

            print(f'[Epoch {epoch + 1:03}/{epochs}]   Train error: {train_error[-1]:.8f}   Val error: {val_error[-1]:.8f}')

            # Early stopping
            if epoch:
                # Compare prev epoch val error with curr
                if val_error[-1] - val_error[-2] >= min_delta:
                    # Reduce patience if error incr >= min_delta
                    curr_patience -= 1
                    if curr_patience == 0:
                        # Ran out of patience, val_error incr too much, stop training
                        print(f'Minimum delta ({min_delta}) exceeded for {patience} consecutive epochs...\n')

                        if restore_best_weights:
                            self.setWeights(best_weights)

                        return (train_error,val_error)

                else:
                    # Reset patience
                    curr_patience = patience

        print('Reached final epoch...\n')
        if restore_best_weights:
            self.setWeights(best_weights)
        
        return (train_error,val_error)

    def __repr__(self):
        return f'Inputs: {self.inputSize} \nOutputs: {self.outputSize}\n' + \
            '\n'.join([str(layer) for layer in self.layers])