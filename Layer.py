from Neuron import Neuron, BiasNeuron, InputNeuron

class Layer:
    def __init__(self):
        self.neurons = []

    @property
    def weights(self):
        return [neuron.getWeights() for neuron in self.neurons]

    def size(self):
        return len(self.neurons)

    def nonBiasNeurons(self):
        return [neuron for neuron in self.neurons if not isinstance(neuron, BiasNeuron)]

    def cleanNeurons(self):
        for neuron in self.neurons:
            neuron.inputVal = 0

    def propagateNeurons(self, lam):
        for neuron in self.neurons:
            neuron.propagate(lam)

    def setWeights(self, neuron_weights):
        for neuron, weights in zip(self.neurons, neuron_weights):
            neuron.setWeights(weights)

    def updateWeights(self, gradient, learning_rate, momentum_rate):
        for neuron in self.neurons:
            neuron.updateWeights(gradient, learning_rate, momentum_rate)
    
    def __repr__(self):
        return ''.join(['-'] * 30) + \
            f'\nLayer size: {self.size()}\n\n' + \
            '\n'.join([str(neuron) for neuron in self.neurons])

class InputLayer(Layer):
    def __init__(self, size, neurons):
        self.neurons = [BiasNeuron(f'X_B', neurons)]
        self.neurons += [InputNeuron(f'X_{i}', neurons) for i in range(size)]

    def setInputs(self, inputs):
        for neuron, val in zip(self.nonBiasNeurons(), inputs):
            neuron.inputVal = val

class HiddenLayer(Layer):
    def __init__(self, size, neurons, num):
        self.neurons = [BiasNeuron(f'H{num}_B', neurons)]
        self.neurons += [Neuron(f'H{num}_{i}', neurons) for i in range(size)]
            

class OutputLayer(Layer):
    def __init__(self, size, useLinear):
        self.neurons = [Neuron(f'Y_{i}') for i in range(size)]
        self.useLinear = useLinear
    
    def getOutput(self):
        if self.useLinear:
            return [neuron.inputVal for neuron in self.neurons]
        else:
            return [neuron.activationVal for neuron in self.neurons]