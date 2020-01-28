#initializing the convolutional neural networkrs
from keras.models import Sequential
classifier = Sequential()

#deprecated method
#classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

#Convolution to generate Feature Maps
from keras.layers.convolutional import Conv2D
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

#Pooling to generate Max Pooled Feature Maps
from keras.layers import MaxPooling2D
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Adding the second Convolutional Layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Flattening
from keras.layers import Flatten
classifier.add(Flatten())

#Full Connection
from keras.layers import Dense

#Hidden layer
classifier.add(Dense(units = 128, activation = 'relu'))

#adding the second hidden layer
classifier.add(Dense(units = 128, activation = 'relu'))

#Output layer
classifier.add(Dense(units = 1, activation = 'sigmoid'))

#Compiling the Convolutional Meural Network
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting the Convolutional Neural Networks to the Images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('C:/ml/Machine Learning A-Z Template Folder/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)/Convolutional_Neural_Networks/dataset/training_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')

test_set = test_datagen.flow_from_directory('C:/ml/Machine Learning A-Z Template Folder/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)/Convolutional_Neural_Networks/dataset/test_set',
                                                        target_size=(64, 64),
                                                        batch_size=32,
                                                        class_mode='binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 5,
                         validation_data = test_set,
                         validation_steps = 2000)

