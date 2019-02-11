import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib.pyplot as plt

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils


from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')


# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense((48 * 48), input_dim=48 * 48,
                    kernel_initializer='normal', activation='relu'))
    model.add(Dense(7, kernel_initializer='normal', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model

def baseline_CNN_model():
	# create model
	model = Sequential()
	model.add(Conv2D(32, (5, 5), input_shape=(1, 48, 48), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(7, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# define the larger model
def larger_model():
	# create model
	model = Sequential()
	model.add(Conv2D(30, (5, 5), input_shape=(1, 48, 48), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(15, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(7, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def getAlexNet():
    #Instantiate an empty model
    model = Sequential()

    # 1st Convolutional Layer
    model.add(Conv2D(filters=96, input_shape=(48,48,1), kernel_size=(11,11), strides=(4,4), padding='valid'))
    model.add(Activation('relu'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))

    # 4th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))

    # 5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

    # Passing it to a Fully Connected layer
    model.add(Flatten())
    # 1st Fully Connected Layer
    model.add(Dense(4096, input_shape=(48*48*1,)))
    model.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.4))

    # 2nd Fully Connected Layer
    model.add(Dense(4096))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))

    # 3rd Fully Connected Layer
    model.add(Dense(1000))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))

    # Output Layer
    model.add(Dense(7))
    model.add(Activation('softmax'))

    # Compile the model
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy']) 

    return model

def getXception():
    keras.applications.xception.Xception(include_top=True, weights=None, input_tensor=None, input_shape=(1, 48, 48), pooling=None, classes=7)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def getVGG16():
    model = keras.applications.vgg16.VGG16(include_top=True, weights=None, input_tensor= None, input_shape=(1, 48, 48), pooling=None, classes=7)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def getRasNet50():
    model = keras.applications.resnet50.ResNet50(include_top=True, weights=None, input_tensor=None, input_shape= (1, 48, 48), pooling=None, classes=7)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def getInception():
    model = keras.applications.inception_v3.InceptionV3(include_top=True, weights=None, input_tensor=None, input_shape=(1, 48, 48), pooling=None, classes=7)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def getDenseNet():
    model = keras.applications.densenet.DenseNet121(include_top=True, weights=None , input_tensor=None, input_shape=(1, 48, 48), pooling=None, classes=7)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def trainNN(train, test):
    X_train = np.array([train[i][1] for i in range(len(train))])
    y_train = np.array([train[i][0] for i in range(len(train))])

    X_test = np.array([test[i][1] for i in range(len(test))])
    y_test = np.array([test[i][0] for i in range(len(test))])

    X_train = X_train.reshape(X_train.shape[0], 48 * 48).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 48 * 48).astype('float32')

    # normalize inputs from 0-255 to 0-1
    X_train = X_train / 255
    X_test = X_test / 255

    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    # build the model
    model = baseline_model()
    # Fit the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
            epochs=100, batch_size=200, verbose=1)
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Baseline Error: %.2f%%" % (100-scores[1]*100))

def trainCNN(train, test):
    X_train = np.array([train[i][1] for i in range(len(train))])
    y_train = np.array([train[i][0] for i in range(len(train))])

    X_test = np.array([test[i][1] for i in range(len(test))])
    y_test = np.array([test[i][0] for i in range(len(test))])

    X_train = X_train.reshape(X_train.shape[0], 1, 48, 48).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 1, 48, 48).astype('float32')

    # normalize inputs from 0-255 to 0-1
    X_train = X_train / 255
    X_test = X_test / 255
    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)


    model = larger_model()
    # model = getVGG16()
    # model.summary()
    # model = getXception()
    # model.summary()
    # model = getRasNet50()
    # model.summary()
    # model = getInception()
    # model.summary()
    # model = getDenseNet()
    # model.summary()

    # Fit the model
    # model = getAlexNet()
    # model.summary()
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=200, verbose=1)
    # # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("CNN Error: %.2f%%" % (100-scores[1]*100))



with open('train', 'rb') as f:
    train = pickle.load(f)

with open('test', 'rb') as f:
    test = pickle.load(f)

#trainNN(train, test)
trainCNN(train, test)
