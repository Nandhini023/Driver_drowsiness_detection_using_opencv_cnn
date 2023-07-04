#importing required libraries
import cv2
from keras.models import load_model
import numpy as np
import time
import urllib.request
import imutils
import warnings
warnings.filterwarnings("ignore")

#for playing alarm sound
import winsound as ws

#set frequency and duration
frequency = 2000
duration = 1000

#url for webcam
url = 'http://192.168.147.247:8080/shot.jpg'

#loading haar cascade face detector file
face = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

#loading haar cascade detector for right and left eye
leye = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')

#labels
lbl=['open eyes','close eyes']

#loading the model
model = load_model('model.h5')

#Initializing the Camera
cam = cv2.VideoCapture(0)

#font
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

#initializing the count, score and thicc values
count=0
score=0
thicc=2

#initializing list values
rpred=[99]
lpred=[99]
r_index = 0
l_index = 0

#Camera
while(True):

    imgPath = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgPath.read()), dtype = np.uint8)
    frame = cv2.imdecode(imgNp, -1)
    frame = imutils.resize(frame, width = 1200)
    height,width = frame.shape[:2]

    #converting to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detecting face in the frame
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))

    #detecting left and right eyes in the detected face
    left_eye = leye.detectMultiScale(gray)
    right_eye =  reye.detectMultiScale(gray)

    #drawing a rectangle
    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

    #drawing a rectangle around the face
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (0,255,0) , 2 )

    #right eye classification
    for (x,y,w,h) in right_eye:
        #cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)
        r_eye=frame[y:y+h,x:x+w]
        count=count+1
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(86,86))
        r_eye= r_eye/255
        r_eye=  r_eye.reshape(86,86,-1)
        r_eye = np.expand_dims(r_eye,axis=0)
        rpred = model.predict(r_eye)
        r_index = np.argmax(rpred)
        if(r_index==1):
            lbl='open eyes' 
        else:
            lbl='close eyes'
        break

    #left eye classification
    for (x,y,w,h) in left_eye:
        #cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        l_eye=frame[y:y+h,x:x+w]
        count=count+1
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
        l_eye = cv2.resize(l_eye,(86,86))
        l_eye= l_eye/255
        l_eye=l_eye.reshape(86,86,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        lpred = model.predict(l_eye)
        l_index = np.argmax(lpred)
        if(l_index == 1):
            lbl='open eyes'   
        else:
            lbl='close eyes'
        break

    #drowsy or not using labels
    if(r_index == 0 and l_index == 0):
        score = score + 1
        cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    # if(rpred[0]==1 or lpred[0]==1):
    else:
        score = 0
        cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    
    if(score < 0):
        score = 0   
    cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    if(score > 3):        #person is feeling sleepy so we beep the alarm        
        try:
            for i in range(3, score):
                frequency += 100
                ws.Beep(2000,1500)
        except:  # isplaying = False
            pass
        if(thicc<16):
            thicc= thicc+4
        else:
            thicc=thicc-2
            if(thicc<2):
                thicc=2
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#closing the camera
cam.release()
cv2.destroyAllWindows()
