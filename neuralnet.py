import numpy as np
from scipy.special import expit
from constants import *

class NeuralNetMLP(object):
	def __init__(self, layers, random_state=None):
		""" Initialise the layers as list(input_layer, ...hidden_layers..., output_layer) """
		np.random.seed(random_state)
		self.num_layers = len(layers)
		self.layers = layers
		self.initialize_weights()

	def initialize_weights(self):
		""" Randomly generate biases and weights for hidden layers. 
		Weights have a Gaussian distribution with mean 0 and
		standard deviation 1 over the square root of the number
		of weights connecting to the same neuron """
		self.biases = [np.random.randn(y, 1) for y in self.layers[1:]]
		self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(self.layers[:-1], self.layers[1:])]

	def fit(self, training_data, l1=0.0, l2=0.0, epochs=500, eta=0.001, minibatches=1, regularization = L2):
		""" Fits the parameters according to training data.
		l1(2) is the L1(2) regularization coefficient. """
		self.l1 = l1
		self.l2 = l2
		n = len(training_data)
		for epoch in xrange(epochs):
			random.shuffle(training_data)
			mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0, n, minibatches)]
			for mini_batch in mini_batches:
				self.batch_update( mini_batch, eta, len(training_data), regularization)

	def batch_update(self, mini_batch, eta, n, regularization=L2):
		""" Update the network's weights and biases by applying gradient
		descent using backpropagation to a single mini batch. """
		nabla_b = [np.zeroes(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		for x, y in mini_batch:
			delta_nabla_b, delta_nabla_w = self.back_propogation(x, y)
			nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
			nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
		self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]
		if regularization == L2:
			self.weights = [(1-eta*(self.l2/n))*w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
		elif regularization == L1:
			self.weights = [w - eta*self.l1*np.sign(w)/n-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]


	def back_propogation(self, x, y, fn = SIGMOID):
		""" Gradient for cost function is calculated from a(L) and 
		back-propogated to the input layer.
		Cross Entropy cost functionis associated with sigmoid neurons, while
		Log-Likelihood cost function is associated with softmax neurons."""
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		activation = x
		activations = [x]
		zs = []    
		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, activation)+b
			zs.append(z)
			if fn == SIGMOID:
				activation = sigmoid(z)
			else:
				activation = softmax(z)
			activations.append(activation)
		dell = delta(activations[-1], y)
		nabla_b[-1] = dell
		nabla_w[-1] = np.dot(dell, activations[-2].transpose())
		for l in xrange(2, self.num_layers -2, 0, -1):
			dell = np.dot(self.weights[l+1].transpose(), dell) * derivative(zs[l], fn)
			nabla_b[-l] = dell
			nabla_w[-l] = np.dot(dell, activations[-l-1].transpose())
		return (nabla_b, nabla_w)

	def cross_entropy_loss(a, y):
		return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

	def log_likelihood_loss(a, y):
		return -np.dot(y, softmax(a).transpose())

	def delta(a, y):
		""" delta for both activations works out to be the same"""
		return (a-y)

	def sigmoid(z):
		""" expit is equivalent to 1.0/(1.0 + np.exp(-z)) """
		return expit(z)

	def softmax(z):
		e = np.exp(float(z))
		return (e/np.sum(e))
	
	def derivative(z, fn):
		""" derivative for f is f(1-f) for respective cost functions """
		if fn == SIGMOID:
			f = sigmoid
		elif fn == SOFTMAX:
			f = softmax
		return f(z)*(1-f(z))