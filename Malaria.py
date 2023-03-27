import logging
import pickle

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import numpy as np
from PIL import Image
import warnings

warnings.filterwarnings("ignore")


class Malaria:
    classifier_pkl_location = "pkls/classifier.pkl"
    prediction_images_location = "static/images/Malaria Cells/single_prediction/"

    def train_model(self):
        try:
            # Initialising the CNN
            classifier = Sequential()

            # Step1 - Convolution
            # Input Layer/dimensions
            # Step-1 Convolution
            # 64 is number of output filters in the convolution
            # 3,3 is filter matrix that will multiply to input_shape=(64,64,3)
            # 64,64 is image size we provide
            # 3 is rgb
            classifier.add(Convolution2D(64, 3, 3, input_shape=(64, 64, 3), activation='relu'))

            # Step2 - Pooling
            # Processing
            # Hidden Layer 1
            # 2,2 matrix rotates, tilts, etc to all the images
            classifier.add(MaxPooling2D(pool_size=(2, 2)))

            # Adding a second convolution layer
            # Hidden Layer 2
            # relu turns negative images to 0
            classifier.add(Convolution2D(64, 3, 3, activation='relu'))
            classifier.add(MaxPooling2D(pool_size=(2, 2)))

            # step3 - Flattening
            # converts the matrix in a singe array
            classifier.add(Flatten())

            # Step4 - Full COnnection
            # 128 is the final layer of outputs & from that 1 will be considered ie dog or cat
            classifier.add(Dense(units=128, activation='relu'))
            classifier.add(Dense(units=1, activation='sigmoid'))
            # sigmoid helps in 0 1 classification

            # Compiling the CNN
            classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            # Deffining the Training and Testing Datasets
            from keras.preprocessing.image import ImageDataGenerator
            train_datagen = ImageDataGenerator(rescale=1. / 255,
                                               shear_range=0.2,
                                               zoom_range=0.2,
                                               horizontal_flip=True)

            test_datagen = ImageDataGenerator(rescale=1. / 255)

            training_set = train_datagen.flow_from_directory(
                'Malaria Cells/training_set',
                target_size=(64, 64),
                batch_size=32,
                class_mode='binary')

            test_set = test_datagen.flow_from_directory(
                'Malaria Cells/testing_set',
                target_size=(64, 64),
                batch_size=32,
                class_mode='binary')

            # nb_epochs how much times you want to back propogate
            # steps_per_epoch it will transfer that many images at 1 time
            # & epochs means 'steps_per_epoch' will repeat that many times
            classifier.fit_generator(
                training_set,
                steps_per_epoch=5000,
                epochs=10,
                validation_data=test_set,
                validation_steps=1000)

            # Saving classifier
            pickle.dump(classifier, open(self.classifier_pkl_location, 'wb'))

            return 1
        except Exception as e:
            logging.ERROR("Tarining Failed")
            return 0

    def predict(self, image_name):
        try:
            # Verifing ouor Model by giving samples of cell to detect malaria
            classifier = pickle.load(open(self.classifier_pkl_location, 'rb'))
            target_iamge = "{0}{1}".format(self.prediction_images_location, image_name)
            test_image = Image.open(target_iamge)
            new_size = (64, 64)
            test_image = test_image.resize(new_size)
            test_image = np.array(test_image)  # Image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            result = classifier.predict(test_image)
            # training_set.class_indices
            if result[0][0] == 1:
                prediction = 'Uninfected'
            else:
                prediction = 'Parasitised'

            return prediction

        except Exception as e:
            logging.ERROR("Tarining Failed")
            return "Failed"



