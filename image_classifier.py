#Import the required packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense

class CNN:

	@staticmethod
	def build(width, height, depth, classes):
		#Construct the model
		model = Sequential()
		inputShape = (height, width, depth)

		#First set of CONV -> RELU -> POOL layers
		model.add(Conv2D(20, (5, 5), padding="same",
			input_shape = inputShape))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		#Second set of CONV -> RELU -> POOL layers
		model.add(Conv2D(50, (5, 5), padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		#Set of FC -> RELU layers
		model.add(Flatten())
		model.add(Dense(500))
		model.add(Activation("relu"))

		#Softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		#Return the constructed architecture
		return model
