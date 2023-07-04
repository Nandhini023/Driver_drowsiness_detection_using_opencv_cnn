import numpy as np
import os
from keras.models import load_model
import tensorflow as tf
import cv2
from sklearn.metrics import classification_report as csr
from sklearn.metrics import accuracy_score as accu
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix as cf
import warnings
warnings.filterwarnings("ignore")

model = load_model('model.h5')

train_dir ='Data/train'
test_dir  ='Data/test'

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
train_generator=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.0,
                                                          rotation_range=7,
                                                          horizontal_flip=True,
                                                          validation_split=0.05
                                                         ).flow_from_directory(train_dir,
                                                                               class_mode = 'categorical',
                                                                               color_mode="grayscale",
                                                                               batch_size = 8,
                                                           target_size=(width,height),
                                                                              subset="training")
validation_generator=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.0,
                                                         ).flow_from_directory(test_dir,
                                                                               class_mode = 'categorical',
                                                                               color_mode="grayscale",
                                                                               batch_size = 8,
                                                                               shuffle = False,
                                                           target_size=(width,height))
validing=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.0,
                                                          rotation_range=7,
                                                          horizontal_flip=True,
                                                         validation_split=0.05
                                                        ).flow_from_directory(train_dir,
                                                                              batch_size = 8,
                                                                              class_mode = 'categorical',
                                                                              color_mode="grayscale",
                                                           target_size=(width,height),subset='validation',shuffle=True)

val = np.array(valSET)

x_val=[]
y_val=[]
for x,y in val:
    if "open" in y.lower():
        y_val.append(1)
    else:
        y_val.append(0)
    x = np.expand_dims(x,axis=2)
    x_val.append(x)

x_val=np.array(x_val)
y_val=np.array(y_val)

y_pd=model.predict(x_val, verbose = 0)

y_pd=np.argmax(y_pd,axis=1)

print("Class Indices:",train_generator.class_indices)
print("Classification Report:",csr(y_val,y_pd))
print("R2 Score:",r2_score(y_val,y_pd))
print("Confusion Matrix:",cf(y_val,y_pd))
print("Accuracy:",accu(y_val,y_pd)*100)
