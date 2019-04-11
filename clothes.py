import keras
from keras import models
from keras import layers
import numpy as np

class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc') > 0.7):
            print("\nReached 60% accuracy, good enough for government work! Bailing...")
            self.model.stop_training = True

callbacks = myCallback()
mnist = keras.datasets.fashion_mnist
(train_imgs, train_labels), (test_imgs, test_labels) = mnist.load_data()
train_imgs = train_imgs/255.0
test_imgs = test_imgs/255.0
model = keras.models.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_imgs, train_labels, epochs=5, callbacks=[callbacks])
