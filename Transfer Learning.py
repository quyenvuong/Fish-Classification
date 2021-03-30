from keras import layers
from keras import models
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg16 import VGG16

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.92):
      print("\nReached 92% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()

train_dir = 'Fish_input/train'
validation_dir = 'Fish_input/Validation'
test_dir = 'Fish_input/test'

pre_train_model = VGG16(input_shape= (224,224,3),
                              include_top= False,
                              weights= 'imagenet')
pre_train_model.summary()
pre_train_model.trainable = False

model = models.Sequential()
model.add(pre_train_model)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(9, activation='softmax'))

model.summary()

from keras import optimizers

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    # rotation_range=30,
    # width_shift_range=0.3,
    # height_shift_range=0.5,
    # shear_range=0.1,
    # zoom_range=0.2,
    # horizontal_flip=True
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224,224),
    class_mode='categorical'
)


train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224,224),
    batch_size=10,
    class_mode='categorical'
)
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(224,224),
    batch_size=10,
    class_mode='categorical'
)
history = model.fit_generator(
    train_generator,
    steps_per_epoch=50,
    epochs=4,
    validation_data=validation_generator,
    validation_steps=50,
    callbacks=[callbacks]
)

score = model.evaluate(test_generator)
model.save('Transfer_Learning_VGG16.h5')

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'bo', label = 'Training acc')
plt.plot(epochs, val_acc, 'b', label = 'Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()