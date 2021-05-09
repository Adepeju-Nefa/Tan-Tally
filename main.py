import cv2
import numpy as np
import face_recognition

picture1=face_recognition.load_image_file('ImagesBasic/billgates.jpg_background=000000&cropx1=292&cropx2=3684&cropy1=592&cropy2=3987')
picture1=cv2.cvtColor(picture1,cv2.COLOR_BGR2RGB)
Testing_Pictures=face_recognition.load_image_file('ImagesBasic/billgates.jpg_background=000000&cropx1=292&cropx2=3684&cropy1=592&cropy2=3987')
Testing_Pictures=cv2.cvtColor(Testing_Pictures,cv2.COLOR_BGR2RGB)

#the actual image
#location of the face
Picture_Location=face_recognition.face_locations(picture1)[0]
Encoded_Picture1=face_recognition.face_encodings(picture1)[0]
#creating the box with the size and color(orange)
cv2.rectangle(picture1,(Picture_Location[3],Picture_Location[0]),(Picture_Location[1],Picture_Location[2]),(51,153,255),5)

#testing the image
faceLocTest=face_recognition.face_locations(Testing_Pictures)[0]
Encoded_Testing=face_recognition.face_encodings(Testing_Pictures)[0]
#creating the box with the size and color(orange)
cv2.rectangle(Testing_Pictures,(Picture_Location[3],Picture_Location[0]),(Picture_Location[1],Picture_Location[2]),(51,153,255),2)

#either true or false
#true if there similar people
#false if theres a difference
answer=face_recognition.compare_faces([Encoded_Picture1],Encoded_Testing)
faceDis=face_recognition.face_distance([Encoded_Picture1],Encoded_Testing)
print(answer,faceDis)
#round the face reconigition to three spaces and a font change and color change
cv2.putText(Testing_Pictures,f'{answer}{round(faceDis[0],3)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(51,153,255),2)


cv2.imshow('Image1:',picture1)
cv2.imshow('Image1:',Testing_Pictures)
cv2.waitKey(0)
