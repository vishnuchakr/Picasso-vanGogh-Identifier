<h1>Picasso versus Vincent van Gogh CNN</h1>
A CNN to classify a painting as created by either Picasso or Vincent van Gogh.

Made with the Keras API using Tensorflow backed.
<h3>image_classifier.py</h3>
This file will define my network architecture for the model. I'll 
implement a LeNet Convolutional Neural Network, modeled by:

INPUT => (CONV => RELU => POOL) x 2 => FC => RELU => FC

LeNet is a simple CNN architecture that is able to perform well on image
datasets, so it is a good fit for a project of this scope. 

I'll be importing a number of packages from Keras:
1. Conv2D: Performs convolution.

2. MaxPooling2D: The pooling layer serves to progressively reduce 
the spatial size of the representation, to reduce the number of 
parameters, memory footprint and amount of computation in the 
network, and hence to also control for over fitting. I will be using
max pooling, but other forms of pooling such as average pooling have
their uses.

3. Activation: I will be using the ReLu activation function instead
of tanh or sigmoid to help prevent vanishing gradients.

4. Flatten: Flattens the network topology into a fully-connected (FC)
layer.

5. Dense: A fully-connected layer.

After the import statements, I will define the CNN class with a build
method. I place the model in its own class for object decomposition purposes, and
give it a static build method to construct the architecture on its own.

The build method takes in a number of parameters:
1. Width: The width of the input images.

2. Height: The height of the input images.

3. Depth: The number of channels of our input image. In this case,
there are 3 (red, green, and blue). 1 would represent grayscale.

4. Classes: The number of classes we want to recognize. In this case, there are two
(Picasso or van Gogh).

<h3>train_network.py</h3>

<h3>validate_network.py</h3>


