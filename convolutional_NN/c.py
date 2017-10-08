# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.utils import np_utils
from keras.utils import load_data
from keras.utils.np_utils import to_categorical
import parser


classifier = Sequential()



classifier.add(Convolution2D(32 , (3,3), input_shape=(64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))




#second layer
classifier.add(Convolution2D(32 , (3, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))



classifier.add(Flatten())

classifier.add(Dense(units = 128, activation = 'relu'))

classifier.add(Dense(units = 3, activation = 'softmax'))



classifier.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)


"""
def to_categorical(y, num_classes=3):
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical
"""

test_set = np_utils.to_categorical(y2, 3)


training_set = train_datagen.flow_from_directory(
        'dataset/training_sets',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

training_set=to_categorical(training_set)

training_set = np_utils.to_categorical(training_set, 3)

test_set = test_datagen.flow_from_directory(
        'dataset/test_sets',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')



classifier.fit_generator(
        training_set,
        steps_per_epoch=1500,
        epochs=1,
        validation_data=test_set,
        validation_steps=600)

'''import numpy as np

from keras.preprocessing import image

test_image = image.load_img('dataset/test/d.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)




rint result
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'''




