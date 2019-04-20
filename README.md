<h1>Picasso versus Vincent van Gogh CNN</h1>
A CNN to classify a painting as created by either Picasso or Vincent van Gogh.

Made with the Keras API using Tensorflow backend.
<h3>image_classifier.py</h3>
![image_classifier](https://user-images.githubusercontent.com/42984263/56462413-630cdc00-6388-11e9-8b31-33e03c324bdb.PNG)
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

On line 14, I construct the model using Sequential() from Keras, since I'm
sequentially adding layers to the CNN.

Line 15 initializes shape of the input using a channels last format,
the default for Tensorflow.

On lines 17 - 26, I add a CONV => RELU => POOL layer two times I'll
refer to these as C,R,P layers.

For the first C,R,P layer, the CONV layer will have 20 convolution filters. I then
apply a ReLu function, followed by 2x2 max pooling in both the x and y directions
with a stride of 2.

For the next C,R,P layer, the CONV layer has 50 convolution filers. It's
common to see the number of CONV filters increase the deeper we go into
the network.

Lines 28 - 38 make up the final block of code in this file.

On line 29, I take the output of the preceding MaxPooling2D layer and
flatten it into a single vector. This allows me to apply my dense/fully-connected layers.
The fully-connected layers consist of 500 nodes (Line 30). On line 31, I
pass this through a final ReLu activation function.

On line 34, I define another fully-connected layer, with the number of nodes
equal to the number of classes that I want to recognize. This dense layer is
given to a softmax classifier, which will yield the probability of each class
being outputted.

Finally, Line 42 returns the fully constructed deep learning + Keras 
image classifier to the calling function.
 
<h3>train_network.py</h3>
This file will train the CNN to classify paintings as either created by
Picasso or van Gogh.

On lines 1 - 18, I import the packages required for this file. These packages
enable me to 
1. Load the image dataset from disk.

2. Pre-process the images.

3. Instantiate the CNN.

4. Train the model.

On line 3, I set the matplotlib backend to 'Agg' so that I can save the 
plot to disk in the background.

From here, I define command line arguments to simplify compilation of the model.



<h3>validate_network.py</h3>


