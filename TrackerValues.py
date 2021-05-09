import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


#import images
path='ImageAttendance'
#creates a list of the images that will be impported
#which will include there names
images=[]
BeachView=[]
People_listnames=os.listdir(path)
print(People_listnames)

#import the classes
#read the current image, our path which is our image
#bill gates.jpg example
#
for values in People_listnames:
    image=cv2.imread(f'{path}/{values}')
    images.append(image)
    BeachView.append(os.path.splitext(values[0]))
print(BeachView)


#list of images
#we will create an empty list with all our encodings
#and we will create a loop which will go through all the encodes

def Search_Encoding(images):
    encodeList=[]
    for values2 in images:
       values2=cv2.cvtColor(values2,cv2.COLOR_BGR2RGB)
       encodevalue=face_recognition.face_encodings(values2)[0]
       encodeList.append(encodevalue)
    return encodeList

def Attendance(name):
    #name and time to keep track
    #we have a lib created in the other folder
    #we wanna read and write that file
    with open('Tally.csv', 'r+') as f:
        myDataList=f.readlines()
        print(myDataList)
        nameList=[]
        #name and time seperated
        for line in myDataList:
            entry=line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now=datetime.now()
            #split by time
            #add first element
            dtString=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

#testing with images
Attendance('Anne')
Attendance('beachboy')
#at the end
List_Values=Search_Encoding(images)
print('Finished')
#images from the camera
webcam_image=cv2.VideoCapture(0)

while True:
    success,imgage_correct=webcam_image.read()
    #pixel size and the scale and then conver to rgb
    imgS=cv2.resize(imgage_correct,(0,0),None,0.30,0.30)
    imgage_correct=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    Webcame_faces=face_recognition.face_locations(imgS)
    Webcame_Encoded=face_recognition.face_encodings(imgS,Webcame_faces)
#trace the faces and compare with all the encodings we found before
#one by one it will check a face from the current frame and grab the encdoings aswell
    for faces,faces_Location in zip(Webcame_Encoded,Webcame_faces):
        similar=face_recognition.compare_faces(List_Values, faces)
        face_Distance=face_recognition.face_distance(List_Values,faces)
        print(face_Distance)
        matching_value=np.argmin(face_Distance)

#convert the name to capital if it is lowercase
        if similar[matching_value]:
            name=BeachView[matching_value].upper()
            print(name)
#create the rectangle, find the location first
#
            vertical1, horizontal2, vertical2, horizontal1 = faces_Location
            vertical1, horizontal2, vertical2, horizontal1 =vertical1*3, horizontal2*3, vertical2*3, horizontal1*3
            cv2.rectangle(imgage_correct, (horizontal1, vertical1), (horizontal2, vertical2), (51,153,255), 2)
            cv2.rectangle(imgage_correct, (horizontal1, vertical2 - 35), (horizontal2, vertical2), (51,153,255), cv2.FILLED)
            cv2.puText(imgage_correct, name, (horizontal1 + 6, vertical2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1(51,153,255), 2)

    cv2.imshow('Webcam',imgage_correct)
    cv2.waitKey(1)