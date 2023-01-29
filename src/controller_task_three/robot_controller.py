import math

import numpy as np
from controller import Controller
from PIL import Image
import cv2

def sigmoid_activation(x):
	return 1./(1.+np.exp(-x))

def tanh_activation(x):
	return np.tanh(x)


# implements controller structure for player
class robotController(Controller):
	def __init__(self, rob):
		# Number of hidden neurons
		self.n_hidden_neurons = 11
		self.n_hidden = [self.n_hidden_neurons]
		self.number_of_sensors = 13
		self.number_of_actions = 2
		self.rob = rob

		self.back_L = 0
		self.back_R = 0
		self.back_C = 0
		self.front_LL = 0
		self.front_L = 0
		self.front_C = 0
		self.front_R = 0
		self.front_RR = 0

		self.green_left = 0
		self.green_center = 0
		self.green_right = 0
		self.red_left = 0
		self.red_center = 0
		self.red_right = 0

		self.food_in_gripper = 0

	def makeStep(self, controller: np.array):
		left, right = self.control(controller)
		self.rob.move(left, right, 500)

	def control(self, controller: np.array):
		inputs = self.getInputValues()
		if self.n_hidden[0]>0:
			# Preparing the weights and biases from the controller of layer 1

			# Biases for the n hidden neurons
			bias1 = controller[:self.n_hidden[0]].reshape(1,self.n_hidden[0])
			# Weights for the connections from the inputs to the hidden nodes
			weights1_slice = len(inputs)*self.n_hidden[0] + self.n_hidden[0]
			weights1 = controller[self.n_hidden[0]:weights1_slice].reshape((len(inputs),self.n_hidden[0]))

			# Outputs activation first layer.
			output1 = tanh_activation(inputs.dot(weights1) + bias1)

			# Preparing the weights and biases from the controller of layer 2
			bias2 = controller[weights1_slice:weights1_slice + self.number_of_actions].reshape(1,self.number_of_actions)
			weights2 = controller[weights1_slice + self.number_of_actions:].reshape((self.n_hidden[0],self.number_of_actions))

			# Outputting activated second layer. Each entry in the output is an action
			output = tanh_activation(output1.dot(weights2)+ bias2)[0]
		else:
			bias = controller[:self.number_of_actions].reshape(1, self.number_of_actions)
			weights = controller[self.number_of_actions:].reshape((len(inputs), self.number_of_actions))

			output = sigmoid_activation(inputs.dot(weights) + bias)[0]

		left_wheel = int(30 * output[0])
		right_wheel = int(30 * output[1])

		return [left_wheel, right_wheel]

	def getInputValues(self):
		values = np.array([
			*self.get_irs_values(),
			*self.detect_green(self.rob.get_image_front()),
			*self.detect_red(self.rob.get_image_front()),
			*self.check_if_food_is_in_gripper(),
		], float)

		return np.nan_to_num(values)

	def get_irs_values(self):
		irs_values = self.rob.read_irs()
		self.back_R = irs_values[0]
		self.back_C = irs_values[1]
		self.back_L = irs_values[2]
		self.front_RR = irs_values[3]
		self.front_R = irs_values[4]
		self.front_C = irs_values[5]
		self.front_L = irs_values[6]
		self.front_LL = irs_values[7]

		# print(max(self.front_RR, self.front_R))
		# print(max(self.front_LL, self.front_L))
		# print('######')
		# print(self.front_C)
		# print('######')
		# print(self.back_C)
		# print(self.back_L)
		# print(self.back_R)
		return [
			max(self.front_RR, self.front_R),
			max(self.front_LL, self.front_L),
			self.front_C,
			self.back_C,
			self.back_L,
			self.back_R
		]

	def check_if_food_is_in_gripper(self):
		irs_values = self.rob.read_irs()
		self.food_in_gripper = 0
		if irs_values[5] > 0.55:
			self.food_in_gripper = 1

		return [self.food_in_gripper]

	# https://www.geeksforgeeks.org/multiple-color-detection-in-real-time-using-python-opencv/?ref=rp
	def detect_green(self, image):

		cv2.imwrite("test_pictures_green.png", image)
		# resize if image is not the correct resolution
		if image.shape[0] != 128 or image.shape[1] != 128:
			im = Image.open("test_pictures_green.png")
			size = 128, 128
			im_resized = im.resize(size, Image.ANTIALIAS)
			im_resized.save("image-128-128-green.png", "PNG")
			image = cv2.imread("image-128-128-green.png", cv2.IMREAD_UNCHANGED)

		# Convert the image in
		# BGR(RGB color space) to
		# HSV(hue-saturation-value)
		# color space
		hsvFrame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

		# Set range for green color and
		# define mask
		green_lower = np.array([25, 52, 72], np.uint8)
		green_upper = np.array([102, 255, 255], np.uint8)
		green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

		# Morphological Transform, Dilation
		# for each color and bitwise_and operator
		# between imageFrame and mask determines
		# to detect only that particular color
		kernal = np.ones((5, 5), "uint8")

		# For green color
		green_mask = cv2.dilate(green_mask, kernal)

		# Creating contour to track green color
		contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		green_best_area = 0
		green_best_x = 0
		green_best_y = 0
		for pic, contour in enumerate(contours):
			area = cv2.contourArea(contour)
			if area > green_best_area and area > 50:
				green_best_area = area

				x, y, w, h = cv2.boundingRect(contour)
				green_best_x = int(x + (w / 2))
				green_best_y = int(y + h)

				image = cv2.rectangle(image, (x, y),
									  (x + w, y + h),
									  (0, 255, 0), 2)

				image = cv2.rectangle(image, (green_best_x, green_best_y),
									  (green_best_x + 2, green_best_y + 2),
									  (0, 0, 255), 2)

		cv2.imwrite("image-128-128-green.png", image)

		self.green_left = 0
		self.green_center = 0
		self.green_right = 0
		# only detect green if there is food in the gripper
		if self.check_if_food_is_in_gripper()[0] == 1:
			if green_best_x < 42:
				# print('object is in top left')
				self.green_left = 1
			if 42 < green_best_x < 84:
				# print('object is in top center')
				self.green_center = 1
			if green_best_x > 84:
				# print('object is in top right')
				self.green_right = 1

		return [
			self.green_left,
			self.green_center,
			self.green_right,
		]

	# https://www.geeksforgeeks.org/multiple-color-detection-in-real-time-using-python-opencv/?ref=rp
	def detect_red(self, image):

		cv2.imwrite("test_pictures_red.png", image)
		# resize if image is not the correct resolution
		if image.shape[0] != 128 or image.shape[1] != 128:
			im = Image.open("test_pictures_red.png")
			size = 128, 128
			im_resized = im.resize(size, Image.ANTIALIAS)
			im_resized.save("image-128-128-red.png", "PNG")
			image = cv2.imread("image-128-128-red.png", cv2.IMREAD_UNCHANGED)

		# Convert the image in
		# BGR(RGB color space) to
		# HSV(hue-saturation-value)
		# color space
		hsvFrame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

		# Set range for red color and
		# define mask
		red_lower = np.array([0, 70, 50], np.uint8)
		red_upper = np.array([180, 255, 255], np.uint8)
		red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

		# Morphological Transform, Dilation
		# for each color and bitwise_and operator
		# between imageFrame and mask determines
		# to detect only that particular color
		kernal = np.ones((5, 5), "uint8")

		# For red color
		red_mask = cv2.dilate(red_mask, kernal)

		# Creating contour to track green color
		contours, red_hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		red_best_area = 0
		red_best_x = 0
		red_best_y = 0
		for pic, contour in enumerate(contours):
			area = cv2.contourArea(contour)
			if area > red_best_area and area > 5:
				red_best_area = area

				x, y, w, h = cv2.boundingRect(contour)
				red_best_x = int(x + (w / 2))
				red_best_y = int(y)

				image = cv2.rectangle(image, (x, y),
									  (x + w, y + h),
									  (0, 0, 255), 2)

				image = cv2.rectangle(image, (red_best_x - 1, red_best_y - 1),
									  (red_best_x + 1, red_best_y + 1),
									  (0, 255, 0), 2)

		cv2.imwrite("image-128-128-red.png", image)

		self.red_left = 0
		self.red_center = 0
		self.red_right = 0
		if red_best_x < 50:
			# print('object is in top left')
			self.red_left = 1
		if 50 <= red_best_x <= 78:
			# print('object is in top center')
			self.red_center = 1
		if red_best_x > 78:
			# print('object is in top right')
			self.red_right = 1

		return [
			self.red_left,
			self.red_center,
			self.red_right,
		]

	def detect_object(self):
		for i in range(8):
			raw_value = np.nan_to_num(self.rob.read_irs())[i]
			if raw_value != 0:
				value = np.log(np.array(raw_value))
				if (value / 10 > -1) and (value / 10 < -0.20):
					return True

		return False

	def getDistance(self):
		p1 = np.array(self.rob.food_position())
		p2 = np.array(self.rob.base_position())

		squared_dist = np.sum((p1 - p2) ** 2, axis=0)
		return np.sqrt(squared_dist)
