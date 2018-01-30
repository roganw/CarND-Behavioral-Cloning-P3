"""
Train model
"""
import csv
import cv2
import numpy as np
from random import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Conv2D, MaxPooling2D, Dropout, Activation
import matplotlib.pyplot as plt


def get_samples_parameter(data_path):
    """
    Read from csv file
    """
    samples = []
    with open(data_path) as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            samples.append(line)
    train, validation = train_test_split(samples, test_size=0.2)
    return train, validation

# # 图像反转，训练集只有一个方向时使用
# image_flipped = np.fliplr(image)
# measurement_flipped = -measurement


def generator(samples, image_path, batch_size=32, correction=0.2):
    """
    return samples through generator
    """
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for sample in batch_samples:
                # center, left, right
                filenames = [image_path + sample[i].split('/')[-1] for i in range(3)]
                images.extend([cv2.imread(name) for name in filenames])
                angle = float(sample[3])
                angles.extend([angle, angle + correction, angle - correction])

            X_train = np.array(images)
            y_train = np.array(angles)
            yield X_train, y_train


def LeNet():
    """
    Build LeNet Model
    """
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Conv2D(6, (5, 5), activation='relu'))
    model.add(MaxPooling2D())
    # model.add(Dropout(0.5))
    model.add(Conv2D(6, (5, 5), activation='relu'))
    model.add(MaxPooling2D())
    # model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))

    return model


def Nvidia():
    """
    Build Nvidia Model
    """
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((65, 20), (0, 0))))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))

    # model.add(MaxPooling2D())
    model.add(Flatten())
    # model.add(Dropout(0.5))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    return model


def train_model(model):
    """
    Train model
    """
    csv_data = './data/driving_log.csv'
    image_dir = './data/IMG/'
    train_samples, validation_samples = get_samples_parameter(csv_data)
    batch_size = 32
    correction = 0.2
    # center, left, right
    steps_per_epoch = len(train_samples * 3) / batch_size
    validation_steps = len(validation_samples * 3) / batch_size
    train_generator = generator(train_samples, image_dir, batch_size=batch_size, correction=correction)
    validation_generator = generator(validation_samples, image_dir, batch_size=batch_size, correction=correction)

    # get model
    model = eval(model)()
    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, validation_data=validation_generator,
        validation_steps=validation_steps, epochs=2, workers=1)

    model.save('model.h5')


if __name__ == '__main__':
    # model_name = 'LeNet'
    model_name = 'Nvidia'
    train_model(model_name)
