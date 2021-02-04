import math
import random

class Neuron:
    def __init__(self, name, connections = []):
        self.name = name
        self.inputVal = 0
        self.activationVal = 0
        self.weights = {neuron: self.initWeight() for neuron in connections}
        self.last_weight_delta = {neuron: 0 for neuron in connections}

    def getWeights(self):
        return [weight for weight in self.weights.values()]

    def initWeight(self):
        return random.uniform(-1, 1)

    def setWeights(self, weight_vals):
        if len(self.weights) != len(weight_vals):
            print(f'Number of connections and weights does not coincide. Expected {len(self.weights)}, got {len(weight_vals)}')
            return

        for neuron, weight in zip(self.weights, weight_vals):
            self.weights[neuron] = weight

    def updateWeights(self, gradients, learning_rate, momentum_rate):
        for neuron, grad in zip(self.weights, gradients):
            delta_weight = grad * self.activationVal * learning_rate + self.last_weight_delta[neuron] * momentum_rate
            
            self.last_weight_delta[neuron] = delta_weight
            self.weights[neuron] += delta_weight

    def activationFunction(self, val, lam):
        return 1 / (1 + math.exp(-val * lam))

    def propagate(self, lam):
        self.activationVal = self.activationFunction(self.inputVal, lam)

        for neuron, weight in self.weights.items():            
            neuron.inputVal += weight * self.activationVal

    def __repr__(self):
        return self.name + '\n' + '\n'.join([' + ' + neu.name + ': ' + str(wgh) for neu, wgh in self.weights.items()])

class InputNeuron(Neuron):
    def propagate(self, lam):
        self.activationVal = self.inputVal
        for neuron, weight in self.weights.items():
            neuron.inputVal += weight * self.inputVal

class BiasNeuron(InputNeuron):
    def propagate(self, lam):
        self.activationVal = 1
        for neuron, weight in self.weights.items():
            neuron.inputVal += weight * 1