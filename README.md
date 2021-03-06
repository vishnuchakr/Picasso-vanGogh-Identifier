<h1>Picasso versus Vincent van Gogh CNN</h1>
A CNN to classify a painting as created by either Picasso or Vincent van Gogh.

Made with the Keras API using Tensorflow backend.
<h3>image_classifier.py</h3>
This file will define my network architecture for the model. I'll 
implement a LeNet Convolutional Neural Network, modeled by:

INPUT => (CONV => RELU => POOL) x 2 => FC => RELU => FC

LeNet is a simple CNN architecture that is able to perform well on image
datasets, so it is a good fit for a project of this scope. It is also easy
to train a LeNet quickly without a powerful GPU, which I don't have quick access to.

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

```python
#Import the required packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
```

After the import statements, I will define the CNN class with a build
method. I place the model in its own class for object decomposition purposes, and
give it a static build method to construct the architecture on its own.

```python
class CNN:

	@staticmethod
	def build(width, height, depth, classes):
		#Construct the model
		model = Sequential()
		inputShape = (height, width, depth)
```

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

```python
		#First set of CONV -> RELU -> POOL layers
		model.add(Conv2D(20, (5, 5), padding="same",
			input_shape = inputShape))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		#Second set of CONV -> RELU -> POOL layers
		model.add(Conv2D(50, (5, 5), padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
```

For the first C,R,P layer, the CONV layer will have 20 convolution filters. I then
apply a ReLu function, followed by 2x2 max pooling in both the x and y directions
with a stride of 2.

For the next C,R,P layer, the CONV layer has 50 convolution filers. It's
common to see the number of CONV filters increase the deeper we go into
the network.

Lines 28 - 38 make up the final block of code in this file.

```python
		#Set of FC -> RELU layers
		model.add(Flatten())
		model.add(Dense(500))
		model.add(Activation("relu"))

		#Softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		#Return the constructed architecture
		return model
```

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

```python
#Use matplotlib to save figures in the background
import matplotlib
matplotlib.use("Agg")

#Import the required packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from image_classifier import CNN
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
```

On lines 1 - 18, I import the packages required for this file. These packages
enable me to 
1. Load the image dataset from disk.

2. Pre-process the images.

3. Instantiate the CNN.

4. Train the model.

On line 3, I set the matplotlib backend to 'Agg' so that I can save the 
plot to disk in the background.

From here, I define command line arguments to simplify compilation of the model.

```python
#Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, 
	help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
ap.add_argument("-p", "--plot", type=str, default="56_1.png",
	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())
```

Here I have two required command line arguments, --dataset  and --model , as well as an optional path to the accuracy/loss chart, --plot .

The --dataset switch should point to the directory containing the images I'll be training our image classifier on, while the --model  switch controls where I'll save the serialized image classifier after it has been trained. If --plot  is left unspecified, it will default to plot.png  in this directory if unspecified.

Next I set some training variables, initialize lists, and gather paths to images.

On Lines 31-33 I define the number of training epochs, initial learning rate, and batch size.

```python
#Initialize the number of epochs, initial learning rate, and batch size
EPOCHS = 25
INIT_LR = 1e-3
BS = 32

#Initialize the data and labels
print("loading images...")
data = []
labels = []

#Obtain and randomly shuffle the image paths
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)
```

Then I initialize data and label lists (Lines 37 and 38). These lists will be responsible for storing the images loaded from disk along with their respective class labels.

From there I grab the paths to our input images followed by shuffling them (Lines 41-43).

Now I'll pre-process the images.

```python
#Loop over input images
for imagePath in imagePaths:
	#Load the image, pre-process it, and store it in the data list
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (28, 28))
	image = img_to_array(image)
	data.append(image)

	#Extract the class label from the path and update the labels list
	label = imagePath.split(os.path.sep)[-2]
	label = 1 if label == "Picasso" else 0
	labels.append(label)
```

This loop loads and resizes each image to a fixed 28×28 pixels, and appends the image array to the data  list, followed by extracting the class label from the imagePath. I'm able to extract the label this way because of the way I implemented the file structure for the images.

Next, I'll scale images and create the training and testing splits:

```python
#Scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float")/ 255.0
labels = np.array(labels)

#Partition the data into training and validation splits
#75% of the data for training, 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, 
	labels, test_size=0.25, random_state=42)

#Convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)
```

We further pre-process our input data by scaling the data points from [0, 255] (the minimum and maximum RGB values of the image) to the range [0, 1] on line 59.

I then perform a training/testing split on the data using 75% of the images for training and 25% for testing (Lines 64 and 65). I also convert labels to vectors on lines 68-9.

Subsequently, I'll perform some data augmentation, enabling me to generate “additional” training data by randomly transforming the input images using the parameters below:

```python
#Construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, 
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, 
	horizontal_flip=True, fill_mode="nearest")
```

Lines 72-4 create an image generator object which performs random rotations, shifts, flips, crops, and sheers on our image dataset. This should allow me to use a smaller dataset and still achieve high results.

I can now move on to the actual training of the model.

```python
#Initialize the model
print("Compiling model...")
model = CNN.build(width=28, height=28, depth=3, classes=2)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, 
	metrics=["accuracy"])

#Train the model
print("Training network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)

#Save the model to disk
print("Saving model...")
model.save(args["model"])
```

I build the CNN along with the Adam optimizer on Lines 78-81. Since this is a two-class classification problem, I'll use binary cross-entropy as the loss function.

Training the network is initiated on Lines 83-7, where model.fit_generator is called , supplying the data augmentation object, training/testing data, and the number of epochs I want it to train for.

Line 91 handles saving the model to disk so I can later use our image classification without having to retrain it.

Finally, I plot the results and see how the deep learning image classifier performed:

```python
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Picasso versus van Gogh")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
```

Using matplotlib, I build the plot and save the plot to disk using the --plot  command line argument which contains the path + filename.

I can run the training file in command line with this command:

![compile](https://user-images.githubusercontent.com/42984263/56463007-ff87ac00-6391-11e9-8eb5-2ff2a9c329e7.PNG)

After training, this is what the Loss and Accuracy plot looks like:

![loss](https://user-images.githubusercontent.com/42984263/56463045-b552fa80-6392-11e9-8f01-ba994867f26b.PNG)

From this plot, I can spot a few issues that might be present with the model. I'll address them when I go to optimize my hyperparameters.

<h3>validate_network.py</h3>

The next step is to evaluate the model on example images not part of the training/testing splits. I took an image off of Google Images for this purpose. I'll be using Picasso's "The Weeping Woman" for the example.

```python
#Import the required packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
```

On lines 2-7 I import the required packages. I can use Keras to load my trained model that's saved to disk.

Next, I parse command line arguments:

```python
#Construct the argument parser and give it the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())
```

The model requires the model and the image to be evaluated as parameters.

Then I can load in an image and pre-process it:

```python
#Load the image
image = cv2.imread(args["image"])
orig = image.copy()
 
#Pre-process the image for classifying
image = cv2.resize(image, (28, 28))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
```

The image is loaded and a copy is made on Lines 18 and 19. The copy allows us to later recall the original image and put our label on it. 

Lines 22-25 handles scaling the image to the range [0, 1], converting it to an array, and addding an extra dimension. Adding an extra dimension to the array via np.expand_dims  allows the image to have the shape (1, width, height, 3). Forgetting to do so results in an error when calling model.predict later.

From there we’ll load the classifier model and make a prediction:

```python
#Load the trained CNN
print("loading network...")
model = load_model(args["model"])
 
#Classify the input image
(vanGogh, Picasso) = model.predict(image)[0]
```

Finally, I can use the prediction to draw on the original image copy and display it to the screen:

```python
#Build the label
label = "Picasso" if Picasso > vanGogh else "vanGogh"
proba = Picasso if Picasso > vanGogh else vanGogh
label = "{}: {:.2f}%".format(label, proba * 100)
 
#Draw the label on the image
output = imutils.resize(orig, width=400)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)
 
#Show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)
```

The label is built on line 35 and the corresponding probability value is chosen on line 36. 

On line 37, the text label to be shown to the user on the image is produced. 

I resize the images to a standard width to ensure it will fit on the screen, and then put the label text on the image (Lines 40-42). 

Finally, on Lines 45, we display the output image until a key has been pressed (Line 46).

Now I can call this file from the terminal and see it in action:

![terminal](https://user-images.githubusercontent.com/42984263/56463293-cb62ba00-6396-11e9-8b04-d8cc982ba449.PNG)

Sure enough, the model is pretty confident that "The Weeping Woman" was created by Picasso!

![output](https://user-images.githubusercontent.com/42984263/56463298-e5040180-6396-11e9-8583-282e9bab5cc0.PNG)

<h3>Optimize a Hyperparameter</h3>

This project uses a number of hyperparameters, any of which I could choose to work with. First, I'll analyze the loss function that the model generated earlier.

![lossspikes](https://user-images.githubusercontent.com/42984263/56463352-01ed0480-6398-11e9-837c-a38943be47fc.PNG)

Taking a look at the validation loss plot, I see a number of unusually large spikes. After researching this, I found that this could either be the result of two causes:
1.  Due to my use of the Adam optimizer from Keras in the training file, there is a chance that the "mini-batches" consist of unlucky data for optimization, inducing these spikes in the cost function. This is a problem with the **batch size** hyperparameter being too small.

2. Spikes in validation loss can also be because of higher learning rates that are updating the model a bit too much after every pass. This is a problem with the **learning rate** hyperparameter being too large.

I'll choose to experiment with the batch size for this project, although I just as easily could have addressed another hyperparameter. I'll be using the method of hyperparameter tuning that I learned from Andrew Ng's Machine Learning course on Coursera, as described by this diagram:

![Process](https://user-images.githubusercontent.com/42984263/56463406-51800000-6399-11e9-9157-583ade44ee7d.PNG)

I have a hypothesis that my batch size is too small, resulting in those spikes in my validation loss. I'll test out some larger batch sizes and go from there. I'll be inputting my data into MS Excel to keep track of my results, with the inputs being the batch size and the outputs being the final value for validation loss. However, I will be keeping note of spikes that occur in the Loss/Accuracy plots. Ill try each input 3 times and average the outputs for each trial and use that as the actual output for that batch size value. Training each model takes only a few minutes, so I have the luxury of multiple trials for the experiment.

![bs32](https://user-images.githubusercontent.com/42984263/56463539-087d7b00-639c-11e9-90c6-8c58d8f3fb86.PNG)

For a batch size of 32, the graphs are spiky, and the final validation loss is 0.5427.

I have a hypothesis that the batch size is too small, so I'll now try a batch size of 32 x 2 = 64.

![bs64](https://user-images.githubusercontent.com/42984263/56463623-927a1380-639d-11e9-8d15-cf15e5a6ffb7.PNG)

![64plot](https://user-images.githubusercontent.com/42984263/56463636-bc333a80-639d-11e9-9565-72161a361bef.PNG)

For a batch size of 64, the graphs are still spiky, maybe even more so. However, the final validation loss has decreased. I'll now try a batch size of 64 x 2 = 128.

![bs128](https://user-images.githubusercontent.com/42984263/56463740-68295580-639f-11e9-88a3-52ad8a96ad07.PNG)

![128plot](https://user-images.githubusercontent.com/42984263/56463736-56e04900-639f-11e9-9498-9ca9a73626ca.PNG)

For a batch size of 128, the graphs are extremely spiky, and the final validation loss has increased very much so. I'll try a batch size thats smaller now, (32 + 64) / 2 = 48.

![bs48](https://user-images.githubusercontent.com/42984263/56463812-1c77ab80-63a1-11e9-99c4-1210d8e1aabc.PNG)

![48plot](https://user-images.githubusercontent.com/42984263/56463805-f8b46580-63a0-11e9-83a7-67c429374793.PNG)

For a batch size of 48, the graphs were less spiky in general and the validation loss was better than the initial one of batch size = 32. However, batch size = 64 still returns the lowest validation loss. Lastly, I'll try a validation loss (48 + 64) / 2 = 56.

![bs56](https://user-images.githubusercontent.com/42984263/56464401-218f2780-63ae-11e9-9674-da7ab2ef280b.PNG)

![56plot](https://user-images.githubusercontent.com/42984263/56464398-10461b00-63ae-11e9-838a-76db70053017.PNG)

For a batch size of 56, the graphs were still spiky, and the validation loss was still not much better. Of the values that I tried, a batch size of 64 looks to be the most optimal value that minimizes validation loss. All of the batch sizes resulted in similar accuracies, so I feel confident in saying that a batch size of 64 looks to be the best bet for this specific model.

It's hard to say, but maybe optimizing the learning rate or the number of epochs might have resulted in a better model.
