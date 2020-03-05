import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.preprocessing.image import  ImageDataGenerator
from keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.applications.vgg16 import preprocess_input


train_data_dir = 'asl-alphabet/asl_alphabet_train/asl_alphabet_train'
test_data_dir = 'asl-alphabet/asl_alphabet_test'

input_shape = 200
batch_size = 64
num_classes = 28
epochs = 50



train_gen = ImageDataGenerator(rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
   preprocessing_function = preprocess_input)

# test_gen = ImageDataGenerator(rescale=1./255)

train_generator = train_gen.flow_from_directory(
    train_data_dir,
    target_size=(input_shape,input_shape),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')

validation_generator = train_gen.flow_from_directory(
    train_data_dir, # same directory as training data
    target_size=(input_shape,input_shape),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(input_shape,input_shape,3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam' , loss='categorical_crossentropy', metrics = ['acc'])

keras_callbacks = [
      EarlyStopping(monitor='val_loss', patience=10, mode='min', min_delta=0.0001),
      ModelCheckpoint('model_checkpoint.h5', monitor='val_loss', save_best_only=True, mode='min')
]

model.summary()

History = model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator,
    validation_steps = validation_generator.samples // batch_size,
    epochs = epochs, callbacks= keras_callbacks, verbose=1)

model.save('model/my_model.h5')
model.save('model.h5')

plt.plot(History.history['acc'])
plt.plot(History.history['val_acc'])
plt.grid(True)
plt.title("Accuracy Graph")
plt.xlabel("Epochs",fontsize=16)
plt.ylabel("Accuracy",fontsize=16)
plt.legend(['train','validation'],loc='lower right')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()



plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.grid(True)
plt.title("Loss Graph")
plt.xlabel("Epochs",fontsize=16)
plt.ylabel("Loss",fontsize=16)
plt.legend(['train','validation'],loc='upper right')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()
