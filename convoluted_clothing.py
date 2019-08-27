import keras
from keras import models
from keras import layers
from keras.optimizers import SGD
import numpy as np

class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc') > 0.9):
            print("\nReached 90% accuracy, good enough for government work! Bailing...")
            self.model.stop_training = True

callbacks = myCallback()
mnist = keras.datasets.fashion_mnist
(train_imgs, train_labels), (test_imgs, test_labels) = mnist.load_data()
train_imgs = train_imgs.reshape(60000, 28, 28, 1)
train_imgs = train_imgs/255.0
test_imgs = test_imgs.reshape(10000, 28, 28, 1)
test_imgs = test_imgs/255.0
model = keras.models.Sequential([
    keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(train_imgs, train_labels, epochs=1, callbacks=[callbacks])
test_loss = model.evaluate(test_imgs, test_labels)
