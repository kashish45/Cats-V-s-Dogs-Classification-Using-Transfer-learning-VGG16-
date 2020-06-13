import numpy as np
import pandas as pd


from keras.layers import Dense,Lambda,Flatten,Dropout,Input
from keras.models import Model,Sequential
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from glob import glob


#Resize All The Images
IMAGE_SIZE=[224,224]

train_path="C:/Users/welcome/Desktop/All About COnvolution Networks/VGG16 Implementation/training_set"
test_path="C:/Users/welcome/Desktop/All About COnvolution Networks/VGG16 Implementation/test_set"


#Using VGG Model

vgg=VGG16(input_shape=IMAGE_SIZE+ [3],weights='imagenet',include_top=False)

#DOnt Train Existing weights


for layer in vgg.layers:
    layer.trainable=False
    
    
folders=glob('training_set/*')    

#Our Layers ( We Can Add More If We want)
x=Flatten()(vgg.output)


prediction=Dense(len(folders),activation='softmax')(x)



#Create A Model Objeect

model=Model(inputs=vgg.input,outputs=prediction)

model.summary()

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy']) 


train_datagen=ImageDataGenerator(rescale=1./255,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True)

test_datagen=ImageDataGenerator(rescale=1./255)   



training_set=train_datagen.flow_from_directory('training_set',
                                               target_size=(224,224),
                                               batch_size=32,
                                               class_mode='categorical')

test_set=test_datagen.flow_from_directory('test_set',
                                          target_size=(224,224),
                                          batch_size=32,
                                          class_mode='categorical')



history=model.fit_generator(training_set,validation_data=test_set,
                            epochs=5,
                            steps_per_epoch=10,
                            validation_steps=10)

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'],label='accuracy')
plt.plot(history.history['val_accuracy'],label='validation Accuracy')


plt.plot(history.history['loss'],label='loss')
plt.plot(history.history['val_loss'],label='validation Loss')


import tensorflow as tf
from keras.models import load_model



model.save('new_model.h5')
