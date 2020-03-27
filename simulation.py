from __future__ import print_function
import ctypes
from keras.datasets import cifar10, cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import keras.backend as K
import random
from copy import copy, deepcopy


K.set_image_dim_ordering('th')
img_size = (3, 32, 32)

def kerasnet(nb_classes):
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same',input_shape=(3, 32, 32)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    return model

#######################################
(X_train, y_train), (X_test, y_test) = cifar100.load_data()
nb_classes = 100
nb_epoch = 20

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# build the network
model = kerasnet(nb_classes)

##Graph - 1##
Index_num = 10
data_for_plotting = numpy.zeros((Index_num-1, 4))
# let's train the model using Adam
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.save_weights('x0.h5')

for index in range(1,Index_num):
    # let's train the model using Adam
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.load_weights('x0.h5')
    # let's first find the small-batch solution
    model.fit(X_train, Y_train,batch_size=5000,nb_epoch=nb_epoch,validation_data=(X_test, Y_test),shuffle=True)

    train_xent, train_acc = model.evaluate(X_train, Y_train,batch_size=5000, verbose=0)
    test_xent, test_acc = model.evaluate(X_test, Y_test,batch_size=5000, verbose=0)
    # parametric plot data collection
    data_for_plotting[index-1, :] = [train_xent,train_acc,test_xent,test_acc] 

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.set_xlabel('index')
ax1.set_ylabel('Cross Entropy', color='b')
ax2.set_ylabel('Accuracy', color='r')
ax1.legend(('Train', 'Test'), loc=0)

ax1.plot(range(1,Index_num), data_for_plotting[:, 2], 'b-')
ax2.plot(range(1,Index_num), data_for_plotting[:, 3], 'r-')

ax1.grid(b=True, which='both')
plt.savefig('Graph1.pdf')


##Graph - 2##
Noise_Precentage = [0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5]
Y_train_currupted =[]
for i in range(len(Noise_Precentage)):
    new = deepcopy(Y_train[:])
    for j in range(len(Y_train[:])):
        if (random.uniform(0,1) < Noise_Precentage[i]):
            np.random.shuffle(new[j])
    Y_train_currupted.append(new)

data_for_plotting = np.zeros((len(Noise_Precentage), 4))
for index in range(len(Noise_Precentage)):
    # let's train the model using Adam
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.load_weights('x0.h5')
    # let's first find the small-batch solution
    model.fit(X_train, Y_train_currupted[index],batch_size=5000,nb_epoch=nb_epoch,validation_data=(X_test, Y_test),shuffle=True)

    train_xent, train_acc = model.evaluate(X_train, Y_train_currupted[index],batch_size=5000, verbose=0)
    test_xent, test_acc = model.evaluate(X_test, Y_test,batch_size=5000, verbose=0)
    # parametric plot data collection
    data_for_plotting[index, :] = [train_xent,train_acc,test_xent,test_acc] 

fig, bx1 = plt.subplots()
bx2 = bx1.twinx()

bx1.set_xlabel('Noise Precentage')
bx1.set_ylabel('Cross Entropy', color='b')
bx2.set_ylabel('Accuracy', color='r')
bx1.legend(('Train', 'Test'), loc=0)

bx1.plot(Noise_Precentage, data_for_plotting[:, 2], 'b-')
bx2.plot(Noise_Precentage, data_for_plotting[:, 3], 'r-')

bx1.grid(b=True, which='both')
plt.savefig('Graph2.pdf')


##Graph - 3##
data_for_plotting = np.zeros((len(Noise_Precentage), 4))
for index in range(len(Noise_Precentage)):
    # let's train the model using Adam
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.load_weights('x0.h5')
    # let's first find the small-batch solution
    model.fit(X_train, Y_train_currupted[index],batch_size=5000,nb_epoch=nb_epoch,validation_data=(X_test, Y_test),shuffle=False)

    train_xent, train_acc = model.evaluate(X_train, Y_train_currupted[index],batch_size=5000, verbose=0)
    test_xent, test_acc = model.evaluate(X_test, Y_test,batch_size=5000, verbose=0)
    # parametric plot data collection
    data_for_plotting[index, :] = [train_xent,train_acc,test_xent,test_acc] 

fig, cx1 = plt.subplots()
cx2 = cx1.twinx()

cx1.set_xlabel('Noise Precentage')
cx1.set_ylabel('Cross Entropy', color='b')
cx2.set_ylabel('Accuracy', color='r')
cx1.legend(('Train', 'Test'), loc=0)

cx1.plot(Noise_Precentage, data_for_plotting[:, 2], 'b-')
cx2.plot(Noise_Precentage, data_for_plotting[:, 3], 'r-')

cx1.grid(b=True, which='both')
plt.savefig('Graph3.pdf')

##Graph - 4##
Batch_size = [1,5,10,20,50,100,250,500,1000,2000,5000]
data_for_plotting = np.zeros((len(Batch_size), 4))
index = 0
for size in Batch_size:
    # let's train the model using sgd
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
    model.load_weights('x0.h5')
    # let's first find the small-batch solution
    model.fit(X_train, Y_train,batch_size=size,nb_epoch=nb_epoch,validation_data=(X_test, Y_test),shuffle=True)

    train_xent, train_acc = model.evaluate(X_train, Y_train,batch_size=size, verbose=0)
    test_xent, test_acc = model.evaluate(X_test, Y_test,batch_size=size, verbose=0)
    # parametric plot data collection
    data_for_plotting[index, :] = [train_xent,train_acc,test_xent,test_acc]
    index = index + 1 

fig, dx1 = plt.subplots()
dx2 = dx1.twinx()

dx1.set_xlabel('Batch size')
dx1.set_ylabel('Cross Entropy', color='b')
dx2.set_ylabel('Accuracy', color='r')
dx1.legend(('Train', 'Test'), loc=0)

dx1.plot(Batch_size, data_for_plotting[:, 2], 'b-')
dx2.plot(Batch_size, data_for_plotting[:, 3], 'r-')

dx1.grid(b=True, which='both')
plt.savefig('Graph4.pdf')