# Face-Recognition-System-for-Home-Security


import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_extractor(img):
    global cropped_face
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if faces is():
        return None

    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face

cap=cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count+=1
        face = cv2.resize(face_extractor(frame),(400,400))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        file_name_path = 'F:/faces/user'+str(count)+'.jpg'

        cv2.imwrite(file_name_path,face)

        cv2.putText(face,str(count),(100,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow('Face Cropper',face)
    else:
        print("face not found")
        pass

    if cv2.waitKey(1)==13 or count==50:
        break
cap.release()
cv2.destroyAllWindows()
print('Collecting Sample Complete!!!')



import cv2
import numpy as np
from os import listdir
import pickle
from os.path import isfile, join

data_path = 'F:/faces/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]


Training_Data, Labels = [], []


for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append( np.asarray( images, dtype=np.uint8))
    Labels.append(i)


Labels = np.asarray(Labels, dtype=np.int32)


model = cv2.face.LBPHFaceRecognizer_create()


print("Model trained sucessefully")


model.train(np.asarray( Training_Data), np.asarray(Labels))
print("Model trained sucessefully")



import cv2
import numpy as np
import time


face_classifier = cv2.CascadeClassifier( 'haarcascade_frontalface_default.xml')

def face_detector(img, size=0.5):
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale( gray, 1.3, 5)
    
    if faces is ():
        return img, []
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    image, face = face_detector(frame)
    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        
        results = model.predict(face)
    
        if results[1] < 500:
            matching = int( 100 * (1 - (results[1])/400) )
            display_string = str(matching) + '% Matching it is User'
        cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (152,255,255), 2)
        
        if matching > 80:
            cv2.putText(image, "Door Unlocked", (200, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Face Recognition', image )
        
        else:
            cv2.putText(image, "Door Locked", (200, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            cv2.imshow('Face Recognition', image )
            
    except:
        cv2.putText(image, "No Face Found", (220, 120) , cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.putText(image, "Door Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Face Recognition', image )
        pass
    
    if cv2.waitKey(1) == 13: 
        break
cap.release()
cv2.destroyAllWindows()


GUI

import cv2
import numpy as np
from tkinter import *
import tkinter.messagebox
from os import listdir
import pickle
from os.path import isfile, join
import time


root=Tk()
root.geometry('700x540')

frame = Frame(root, relief=RIDGE, borderwidth=2)
frame.pack(fill=BOTH,expand=1)
root.title('Driver Cam')
frame.config(background='light blue')
label = Label(frame, text="Face Lock",bg='light blue',font=('Times 35 bold'))
label.pack(side=TOP)
filename = PhotoImage(file="C:/Users/Abhishek/Desktop/back.png")
background_label = Label(frame,image=filename)
background_label.pack(side=TOP)

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def datarecord():
    def face_extractor(img):
        global cropped_face
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.3,5)

        if faces is():
            return None

        for(x,y,w,h) in faces:
            cropped_face = img[y:y+h, x:x+w]

        return cropped_face

    cap=cv2.VideoCapture(0)
    count = 0

    while True:
        ret,frame = cap.read()
        if face_extractor(frame) is not None:
            count+=1
            face = cv2.resize(face_extractor(frame),(400,400))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            file_name_path = 'F:/faces/user'+str(count)+'.jpg'

            cv2.imwrite(file_name_path,face)

            cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.imshow('Face Cropper',face)
        else:
            print("face not found")
            pass

        if cv2.waitKey(1)==13 or count==50:
            break
    cap.release()
    cv2.destroyAllWindows()


print('Collecting Sample Complete!!!')


def modeltrain():
    data_path = 'F:/faces/'
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

    
    Training_Data, Labels = [], []


    for i, files in enumerate(onlyfiles):
        image_path = data_path + onlyfiles[i]
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        Training_Data.append( np.asarray( images, dtype=np.uint8))
        Labels.append(i)


    Labels = np.asarray(Labels, dtype=np.int32)


    model = cv2.face.LBPHFaceRecognizer_create()


    print("Model trained sucessefully")


    model.train(np.asarray( Training_Data), np.asarray(Labels))
    print("Model trained sucessefully")
    
    


def test():
    def face_detector(img, size=0.5):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale( gray, 1.3, 5)
    
        if faces is ():
             return img, []
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,255),2)
            roi = img[y:y+h, x:x+w]
            roi = cv2.resize(roi, (200, 200))
        return img, roi
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        image, face = face_detector(frame)
        try:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            results = model.predict(face)
            if results[1] < 500:
                matching = int( 100 * (1 - (results[1])/400) )
                display_string = str(matching) + '% Matching it is User'
            cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (152,255,255), 2)
        
            if matching > 75:
                cv2.putText(image, "Door Unlocked", (200, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                cv2.imshow('Face Recognition', image )
        
            else:
                cv2.putText(image, "Door Locked", (200, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
                cv2.imshow('Face Recognition', image )
            
        except:
            cv2.putText(image, "No Face Found", (220, 120) , cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            cv2.putText(image, "Door Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            cv2.imshow('Face Recognition', image )
            pass
    
        if cv2.waitKey(1) == 13: 
            break
    cap.release()
    cv2.destroyAllWindows()
    
    
    
    

but1=Button(frame,padx=5,pady=5,width=39,bg='yellow',fg='black',relief=GROOVE,command=datarecord,text='Data Record',font=('helvetica 15 bold'))
but1.place(x=100,y=104)  

but1=Button(frame,padx=5,pady=5,width=39,bg='yellow',fg='black',relief=GROOVE,command=modeltrain,text='Model train',font=('helvetica 15 bold'))
but1.place(x=100,y=176)  


but1=Button(frame,padx=5,pady=5,width=39,bg='yellow',fg='black',relief=GROOVE,command=test,text='Model test',font=('helvetica 15 bold'))
but1.place(x=100,y=250)  

root.mainloop()    

