from __future__ import print_function
import ctypes
from keras.datasets import cifar10, cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
import numpy
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K

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
nb_epoch = 2

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# build the network
model = kerasnet(nb_classes)

# let's train the model using Adam
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.save_weights('x0.h5')

#model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

# let's first find the small-batch solution
model.fit(X_train, Y_train,
          batch_size=256,
          nb_epoch=nb_epoch,
          validation_data=(X_test, Y_test),
          shuffle=True)
          
sb_solution = model

# re-compiling to reset the optimizer accumulators
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# setting the initial (starting) point
model.load_weights('x0.h5')

# now, let's train the large-batch solution
model.fit(X_train, Y_train,
          batch_size=5000,
          nb_epoch=nb_epoch,
          validation_data=(X_test, Y_test))
lb_solution = model
# parametric plot data collection
# we discretize the interval [-1,2] into 25 pieces
alpha_range = numpy.linspace(-1, 2, 25)
data_for_plotting = numpy.zeros((25, 4))

train_xent, train_acc = lb_solution = model.evaluate(X_train, Y_train,batch_size=5000, verbose=0)
test_xent, test_acc = lb_solution.evaluate(X_test, Y_test,batch_size=5000, verbose=0)
train_xent2, train_acc2 = sb_solution = model.evaluate(X_train, Y_train,batch_size=5000, verbose=0)
test_xent2, test_acc2 = sb_solution.evaluate(X_test, Y_test,batch_size=5000, verbose=0)

i = 0
for alpha in alpha_range:

    data_for_plotting[i, :] = [train_xent*alpha + train_xent2*(1-alpha), train_acc*alpha + train_acc2*(1-alpha), test_xent*alpha + test_xent2*(1-alpha), test_acc*alpha + test_acc2*(1-alpha)]
    i += 1

# finally, let's plot the data
# we plot the XENT loss on the left Y-axis
# and accuracy on the right Y-axis
# if you don't have Matplotlib, simply print
# data_for_plotting to file and use a different plotter

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(alpha_range, data_for_plotting[:, 0], 'b-')
ax1.plot(alpha_range, data_for_plotting[:, 2], 'b--')

ax2.plot(alpha_range, data_for_plotting[:, 1]*100., 'r-')
ax2.plot(alpha_range, data_for_plotting[:, 3]*100., 'r--')

ax1.set_xlabel('alpha')
ax1.set_ylabel('Cross Entropy', color='b')
ax2.set_ylabel('Accuracy', color='r')
ax1.legend(('Train', 'Test'), loc=0)

ax1.grid(b=True, which='both')
plt.savefig('Figures.pdf')



