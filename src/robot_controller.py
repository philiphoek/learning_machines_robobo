import numpy as np
from controller import Controller

def sigmoid_activation(x):
	return 1./(1.+np.exp(-x))


# implements controller structure for player
class robotController(Controller):
	def __init__(self, _n_hidden):
		# Number of hidden neurons
		self.n_hidden = [_n_hidden]

	def control(self, inputs: np.array, controller: np.array):
		# Normalises the input using min-max scaling
		inputs = (inputs-min(inputs))/float((max(inputs)-min(inputs)))
		print(inputs)

		if self.n_hidden[0]>0:
			# Preparing the weights and biases from the controller of layer 1

			# Biases for the n hidden neurons
			bias1 = controller[:self.n_hidden[0]].reshape(1,self.n_hidden[0])
			print('bias1')
			print(bias1)
			# Weights for the connections from the inputs to the hidden nodes
			weights1_slice = len(inputs)*self.n_hidden[0] + self.n_hidden[0]
			print('weights1_slice')
			print(weights1_slice)
			weights1 = controller[self.n_hidden[0]:weights1_slice].reshape((len(inputs),self.n_hidden[0]))
			print('weights1')
			print(weights1)

			# Outputs activation first layer.
			output1 = sigmoid_activation(inputs.dot(weights1) + bias1)
			print('output1')
			print(output1)

			# Preparing the weights and biases from the controller of layer 2
			bias2 = controller[weights1_slice:weights1_slice + 2].reshape(1,2)
			print('bias2')
			print(bias2)
			weights2 = controller[weights1_slice + 2:].reshape((self.n_hidden[0],2))
			print('weights2')
			print(weights2)

			# Outputting activated second layer. Each entry in the output is an action
			output = sigmoid_activation(output1.dot(weights2)+ bias2)[0]
			print('output')
			print(output)
		else:
			bias = controller[:2].reshape(1, 2)
			weights = controller[2:].reshape((len(inputs), 2))

			output = sigmoid_activation(inputs.dot(weights) + bias)[0]

		# takes decisions about sprite actions
		if output[0] > 0.5:
			left_wheel = 0
		else:
			left_wheel = 10

		if output[1] > 0.5:
			right_wheel = 0
		else:
			right_wheel = 10

		return [left_wheel, right_wheel]
