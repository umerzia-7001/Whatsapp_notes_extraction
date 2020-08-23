
from keras.preprocessing.image import *
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from glob import glob
from model import cnn_model
from keras import *

# Data augmentation

batch_size=4
train_datagen=ImageDataGenerator(rescale=1./255,
                                 rotation_range=40,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True,
                                 fill_mode='nearest')

train_generator=train_datagen.flow_from_directory(
    'data',
    target_size=(124,124),# resizing images
    batch_size=batch_size,
    class_mode='binary'
# since we use binary_crossentropy loss, we need binary labels
)

x,y=next(train_generator)
print(x.shape,y.shape)

# training model

model=cnn_model()
model.fit_generator(
    train_generator,steps_per_epoch=2000//batch_size,
    epochs=5
)
model.save_weights('weights.h5') #saving weights

x,y=next(train_generator)
y=y.reshape(len(y),1)

y_pred=model.predict(x)
y_pred=(y_pred>0.5)*1

y==y_pred


img_path=random.choice(glob('data/1/*'))
img=load_img(img_path,target_size=(124,124,3))
x=img_to_array(img) /255.0
y=model.predict(np.expand_dims(x,axis=0))
print(np.squeeze(y)>0.5)
print(img)