import matplotlib.pyplot as plt
import keras
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras import optimizers
from keras.callbacks import EarlyStopping,ModelCheckpoint, ReduceLROnPlateau
from keras_vggface.vggface import VGGFace
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.engine import Model
from keras.layers import Input
import tensorflow as tf
import cv2
from tqdm import tqdm
import os
import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.metrics import confusion_matrix
from keras.utils import plot_model
from keras.preprocessing import image


#
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# tf.keras.backend.set_session(tf.Session(config=config))


# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


train_data_dir = 'age_dataset_2'

input_shape = 256
batch_size = 1
num_classes = 8
epochs = 1
# num_of_test_samples = 3306

def random_crop(img, random_crop_size):
    # Note: image_data_format is 'channel_last'
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y+dy), x:(x+dx), :]


def crop_generator(batches, crop_length):
    """Take as input a Keras ImageGen (Iterator) and generate random
    crops from the image batches generated by the original iterator.
    """
    while True:
        batch_x, batch_y = next(batches)
        batch_crops = np.zeros((batch_x.shape[0], crop_length, crop_length, 3))
        for i in range(batch_x.shape[0]):
            batch_crops[i] = random_crop(batch_x[i], (crop_length, crop_length))
        yield (batch_crops, batch_y)

train_gen = ImageDataGenerator(
    rescale = 1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
   )


train_batches = train_gen.flow_from_directory(
    train_data_dir,
    target_size=(input_shape, input_shape),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle = True)

validation_batches = train_gen.flow_from_directory(
    train_data_dir, # same directory as training data
    target_size=(input_shape,input_shape),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False)

train_crops = crop_generator(train_batches, 224)
validation_crops = crop_generator(validation_batches, 224)
print(train_batches)

# model1 = VGGFace(model='vgg16')
#
# for layer in model1.layers:
#     if 'conv' not in layer.name:
#         continue
#     filters, biases = layer.get_weights()
#     f_max, f_min = filters.max(), filters.min()
#
#
#     filters = (filters - f_min) / (f_max - f_min)
#
#
#     model = Model(inputs=model1.inputs, outputs=layer.output)
#
#     # my_img = image.load_img('baby.jpeg', target_size=(224, 224, 3))
#     my_img = image.img_to_array()
#     img = np.expand_dims(my_img, axis=0)
#
#     img = preprocess_input(img)
#
#     feature_map = model.predict(img)
#
#     squares, ix = 8, 1
#
#     for _ in range(squares):
#         for _ in range(squares):
#             ax = plt.subplot(squares, squares, ix)
#             ax.set_xticks([])
#             ax.set_yticks([])
#             plt.imshow(feature_map[0, :, :, ix - 1], cmap='gray')
#             ix += 1
#     plt.show()

# filters1, biases1 = model1.layers[1].get_weights()
# print("Single filters")





















# model.add(Dense(1000,activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(100,activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(8,activation='softmax'))
# print(model.summary())

# model.layers[0].trainable = False

# # for i in range(len(vgg_features.layers) - 3):
# #     model.add(vgg_features.layers[i])
# #
# # for layer in vgg_features.layers[:-4]:
# #     layer.trainable = False
#

# layers_output = [layer.output for layer in model.layers[:10]]
#
#
# model.compile(optimizer=optimizers.Adam(lr=0.001), loss='categorical_crossentropy',
#               metrics=['acc'])
#
# checkpoint = ModelCheckpoint('weight_resnet.hdf5', monitor='val_loss', save_best_only=True)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
#                               patience=3, min_lr=0.00001, verbose=1)
# early_stopping_monitor = EarlyStopping(patience=30, verbose=1)
#
# training = model.fit_generator(train_crops, validation_data=validation_crops,
#                                steps_per_epoch = train_batches.samples // batch_size, epochs=epochs,use_multiprocessing= False,
#                                validation_steps = validation_batches.samples // batch_size,
#                                callbacks=[checkpoint, early_stopping_monitor])
#
# model.save('model_age_resnet.h5')
#
#
#
#
# plt.plot(training.history['acc'])
# plt.xlabel(['Epochcount'])
# plt.plot(training.history['val_acc'])
# plt.ylabel(['Accuracy'])
# plt.show()
# plt.savefig('accuracy.png')
#
#
# plt.plot(training.history['loss'])
# plt.xlabel(['Epochcount'])
# plt.plot(training.history['val_loss'])
# plt.ylabel(['Lossdata'])
# plt.show()
# plt.savefig('loss.png')







