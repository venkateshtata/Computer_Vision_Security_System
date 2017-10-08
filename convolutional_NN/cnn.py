# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.utils import np_utils





classifier = Sequential()


#input layer
classifier.add(Conv2D(32 , (3,3), input_shape=(64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))




#second layer
classifier.add(Conv2D(32 , (3, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

#third layer
classifier.add(Conv2D(32 , (3, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

#fourth layer
classifier.add(Conv2D(32 , (3, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

#fifth layer
classifier.add(Conv2D(32 , (3, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))


#sixth layer
classifier.add(Conv2D(32 , (3, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

#seventh layer
classifier.add(Conv2D(32 , (3, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))




classifier.add(Flatten())

classifier.add(Dense(units = 128, activation = 'relu'))
#output layer
classifier.add(Dense(units = 3, activation = 'softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
classifier.compile(loss='categorical_crossentropy', optimizer=sgd, metrics = ['accuracy'])

#classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)



training_set = train_datagen.flow_from_directory(
        'dataset/training_sets',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')



test_set = test_datagen.flow_from_directory(
        'dataset/test_sets',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')


#training_set= to_categorical(training_set, 3)

#test_set =to_categorical(test_set, 3)


classifier.fit_generator(
        training_set,
        steps_per_epoch=150,
        epochs=1,
        validation_data=test_set,
        validation_steps=600)

import numpy as np

from keras.preprocessing import image

test_image = image.load_img('dataset/test/d.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices




'''print result
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'''




