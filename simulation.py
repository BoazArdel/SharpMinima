from __future__ import print_function
import ctypes

hllDll = ctypes.WinDLL("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\bin\\cudart64_101.dll")
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


img_size = (3, 32, 32)
tf.config.experimental.list_physical_devices('GPU')

def kerasnet(nb_classes):
        model = Sequential()
        model.add(Convolution2D(32, 3, 3, border_mode='valid',
                                input_shape=(3,32,32)))
        model.add(BatchNormalization(mode=0,axis=1))
        model.add(Activation('relu'))
        model.add(Convolution2D(32, 3, 3))
        model.add(BatchNormalization(mode=0,axis=1))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(64, 3, 3, border_mode='valid'))
        model.add(BatchNormalization(mode=0,axis=1))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3))
        model.add(BatchNormalization(mode=0,axis=1))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(BatchNormalization(mode=0))
        model.add(Activation('relu'))
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

# let's train the model using Adam
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.save_weights('x0.h5')

# let's first find the small-batch solution
model.fit(X_train, Y_train,
          batch_size=256,
          nb_epoch=nb_epoch,
          validation_data=(X_test, Y_test),
          shuffle=True)
sb_solution = [p.get_value() for p in model.trainable_weights]

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
lb_solution = [p.get_value() for p in model.trainable_weights]

# parametric plot data collection
# we discretize the interval [-1,2] into 25 pieces
alpha_range = numpy.linspace(-1, 2, 25)
data_for_plotting = numpy.zeros((25, 4))

i = 0
for alpha in alpha_range:
    for p in range(len(sb_solution)):
        model.trainable_weights[p].set_value(lb_solution[p]*alpha +
                                             sb_solution[p]*(1-alpha))
    train_xent, train_acc = model.evaluate(X_train, Y_train,
                                           batch_size=5000, verbose=0)
    test_xent, test_acc = model.evaluate(X_test, Y_test,
                                         batch_size=5000, verbose=0)
    data_for_plotting[i, :] = [train_xent, train_acc, test_xent, test_acc]
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



