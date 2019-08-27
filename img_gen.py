import numpy as np
from google.colab import files
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop

uploaded = files.upload()

train_datagen = ImageDataGenerator(rescale=1./255) # rescale normalizes data

traingen = train_datagen.flow_from_directory(
        train_dir, #target dir
        target_size=(300, 300), # this resizes images as they're loaded
        batch_size=128,
        class_mode='binary')

val_datagen = ImageDataGenerator(rescale=1./255) # rescale normalizes data

valgen = val_datagen.flow_from_directory(
        val_dir, #target dir
        target_size=(300, 300), # this resizes images as they're loaded
        batch_size=32,
        class_mode='binary')

model.compile(loss='binary_crossentropy',
        optimizer=RMSprop(lr=0.001), # RMSprop allows for adjustment of learning rate
        metrics=['acc'])

#training
history = model.fit_generator(
        train_generator,
        steps_per_epoch=8,
        epochs=15,
        validation_data=valgen,
        validation_steps=8,
        verbose=2)

for fn in uploaded.keys():
    #predicting images
    path = '/content/' + fn
    img = image.load_img(path, target_size=(300,300))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    print(classes[0])
    if classes[0]>0.5:
        print(fn + "is humanoid")
    else:
        print(fn + "is equine")
