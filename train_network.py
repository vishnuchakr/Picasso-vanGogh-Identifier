#Use matplotlib to save figures in the background
import matplotlib
matplotlib.use("Agg")

#Import the required packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from pyimagesearch.lenet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os

#Construct the argument parser and parse the arguments
ap = argparse.ArgumentParse()
ap.add_argument("-d", "--dataset", required=True, 
	help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

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

#Construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, 
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, 
	horizontal_flip=True, fill_mode="nearest")

#Initialize the model
print("Compiling model...")
model = LeNet.build(width=28, height=28, depth=3, classes=2)
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

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Santa/Not Santa")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])