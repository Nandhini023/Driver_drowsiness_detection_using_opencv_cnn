import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Input, Dropout,Flatten, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau,EarlyStopping 
from tensorflow.keras.utils import plot_model
from IPython.display import SVG, Image
from livelossplot import PlotLossesKerasTF
import tensorflow as tf
import cv2

train_dir ='/Data/train'
test_dir  ='/Data/test'

valSET = []
for feature in os.listdir("Data/"):
    f_list=os.listdir("Data/" + feature)
    print(str(len(os.listdir("Data/" + feature))) + " " + feature + " detected")
    for state in f_list:
        print("\t",str(len(os.listdir("Data/" + feature+"/"+state))) + " " + state + " images")
        for img in os.listdir("Data/" + feature+"/"+state):
            if feature=="test":
                a="Data/" + feature+"/"+state+"/"+img
                x=cv2.resize(cv2.imread(a),(86,86))
                x = cv2.cvtColor(x,cv2.COLOR_BGR2GRAY)
                y=state
                valSET.append((x,y))

width, height = 86, 86
img_size = 86
train_generator=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.0,rotation_range=7,horizontal_flip=True,validation_split=0.05
                                                         ).flow_from_directory(train_dir,class_mode = 'categorical',color_mode="grayscale",batch_size = 8,
                                                           target_size=(width,height),subset="training")
validation_generator=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.0,).flow_from_directory(test_dir,class_mode = 'categorical',
                                                                               color_mode="grayscale",batch_size = 8,shuffle = False,target_size=(width,height))
validing=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.0,rotation_range=7,horizontal_flip=True,validation_split=0.05
                                                        ).flow_from_directory(train_dir,batch_size = 8,class_mode = 'categorical',color_mode="grayscale",
                                                           target_size=(width,height),subset='validation',shuffle=True)

from keras.models import Sequential ,Model
from keras.layers import Dense ,Flatten ,Conv2D ,MaxPooling2D ,Dropout ,BatchNormalization  ,Activation ,GlobalMaxPooling2D
from keras.optimizers import Adam 
from keras.callbacks import EarlyStopping ,ReduceLROnPlateau

optimizer=Adam(lr=0.001,beta_1=0.9,beta_2=0.99,decay=0.001/32)
EarlyStop=EarlyStopping(patience=10,restore_best_weights=True)
Reduce_LR=ReduceLROnPlateau(monitor='val_accuracy',verbose=2,factor=0.5,min_lr=0.00001)
callback=[EarlyStop , Reduce_LR]

num_classes = 2
num_detectors = 32

network = Sequential()

network.add(Conv2D(num_detectors, (3,3), activation='relu', padding = 'same', input_shape = (img_size, img_size, 1)))
network.add(BatchNormalization())
network.add(Conv2D(num_detectors, (3,3), activation='relu', padding = 'same'))
network.add(BatchNormalization())
network.add(MaxPooling2D(pool_size=(2,2)))
network.add(Dropout(0.2))

network.add(Conv2D(2*num_detectors, (3,3), activation='relu', padding = 'same'))
network.add(BatchNormalization())
network.add(Conv2D(2*num_detectors, (3,3), activation='relu', padding = 'same'))
network.add(BatchNormalization())
network.add(MaxPooling2D(pool_size=(2,2)))
network.add(Dropout(0.2))

network.add(Conv2D(2*2*num_detectors, (3,3), activation='relu', padding = 'same'))
network.add(BatchNormalization())
network.add(Conv2D(2*2*num_detectors, (3,3), activation='relu', padding = 'same'))
network.add(BatchNormalization())
network.add(MaxPooling2D(pool_size=(2,2)))
network.add(Dropout(0.2))

network.add(Conv2D(2*2*2*num_detectors, (3,3), activation='relu', padding = 'same'))
network.add(BatchNormalization())
network.add(Conv2D(2*2*2*num_detectors, (3,3), activation='relu', padding = 'same'))
network.add(BatchNormalization())
network.add(MaxPooling2D(pool_size=(2,2)))
network.add(Dropout(0.2))

network.add(Flatten())

network.add(Dense(2 * num_detectors, activation='relu'))
network.add(BatchNormalization())
network.add(Dropout(0.2))

network.add(Dense(2 * num_detectors, activation='relu'))
network.add(BatchNormalization())
network.add(Dropout(0.2))

network.add(Dense(num_classes, activation='softmax'))

network.compile(optimizer=optimizer,loss='categorical_crossentropy', metrics=["accuracy"])

epochs = 30
steps_per_epoch = train_generator.n//train_generator.batch_size
validation_steps = validation_generator.n//validation_generator.batch_size

reduce_lr = ReduceLROnPlateau(monitor='val_accuracy',verbose=2,factor=0.5,
                              patience=2, min_lr=0.00001, mode='auto')
checkpoint = ModelCheckpoint("Eyes.h5", monitor='val_accuracy',
                             save_weights_only=True, mode='max', verbose=2)
callbacks = [EarlyStop ,PlotLossesKerasTF(), checkpoint, reduce_lr]

history = network.fit(
    x=train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data = validation_generator,
    validation_steps = validation_steps,
    callbacks=callbacks
)
history.save('model.h5')
