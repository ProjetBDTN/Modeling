######################################################################################################################################
##                                                     Digital identification                                                       ##
##                                                          LIU Jixiong                                                             ##
##                                                         Version Keras                                                            ##
######################################################################################################################################

# I'm using the type of CNN 

#   In a convolutional neural network, a convolutional layer can have multiple different convolution kernels (also known as filters), 
#and each convolution kernel slides over the input image and processes only a small number of images at a time. Such a convolutional 
#layer at the input can extract the most basic features in the image.
"
#   Each neuron is only connected to a local area of the upper layer. The spatial size of the connection is called the receptive field 
#of the neuron.
# Also the current layer uses the same weight and bias for each channel's neurons in the depth direction, witch called weight sharing.
#
#   Local connections and weight sharing reduce the amount of parameters, greatly reducing training complexity and reducing overfitting. 
#At the same time, weight sharing also gives the convolution network tolerance to translation. So it seems that CNN is a great choice 
#for image identification.

# inconvenient:
#Not easy to understand 
#Can't observe the revolution in the hidden layer
#Need a large calculate power

# import the packages wich will be used 
import pandas as pd
import numpy as np
import keras.layers.core as core
import keras.layers.convolutional as conv
import keras.models as models
import keras.utils.np_utils as kutils

# read the data set from the impot
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

train.head()

# setting and normalizing of coeffients whitch will be used 
epoch = 1

batch_size = 128
rows, cols = 28, 28

# reshape the training data
trainX = train.iloc[:, 1:].values.reshape(train.shape[0], rows, cols, 1)
trainX = trainX.astype(float)

# Normalize the data
trainX /= 255.0

trainY = kutils.to_categorical(train.iloc[:, 0])
nb_classes = trainY.shape[1]


# declaration of the type of model
cnn = models.Sequential()

#layer input and two hidden layers and an output 
# [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out 
cnn.add(conv.Convolution2D(32, 3, 3,  activation="relu", input_shape=(28, 28, 1), border_mode='same'))
cnn.add(conv.Convolution2D(32, 3, 3, activation="relu", border_mode='same'))
#Max pooling operation for temporal data.
cnn.add(conv.MaxPooling2D(strides=(2,2)))

cnn.add(conv.Convolution2D(64, 3, 3, activation="relu", border_mode='same'))
cnn.add(conv.Convolution2D(64, 3, 3, activation="relu", border_mode='same'))
#Max pooling operation for temporal data.
cnn.add(conv.MaxPooling2D(strides=(2,2)))


cnn.add(core.Flatten())
cnn.add(core.Dropout(0.2))
cnn.add(core.Dense(128, activation="relu")) 
cnn.add(core.Dense(nb_classes, activation="softmax"))

#See the summary and complie the networrk
cnn.summary()
cnn.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


# Fitting the data 
cnn.fit(trainX, trainY, batch_size=batch_size, nb_epoch=epoch, verbose=1)

# Resharpe the test set and normalize the data
testX = test.values.reshape(test.shape[0], 28, 28, 1)
testX = testX.astype(float)
testX /= 255.0

# Do the prediction 
yPred = cnn.predict_classes(testX)

test.insert(0,'label', yPred)
test.head()

# Save the prediction 
np.savetxt('mnist-vggnet.csv', np.c_[range(1,len(yPred)+1),yPred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')

