import cv2
import numpy as np
import face_recognition

import os


from datetime import datetime

path='StudentDetails'
images =[]
classNames =[]
mylist = os.listdir(path)
print(mylist)

for cl in mylist:
    curImg=cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList=[]
    for img in images:
        img =cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode =face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('attendance.csv','r+') as f:
        myDataList=f.readlines()
        nameList=[]
        #print(mylist)
        for line in myDataList:
            entry=line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now=datetime.now()
            dtstring=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtstring}')



encodeList = findEncodings(images)
# print(len(encodeList))
print("Encoding complete")

cap = cv2.VideoCapture(0)

while True:
    success,img = cap.read()
    imgs=cv2.resize(img,(0,0),None,0.25,0.25)
    imgs=cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)

    faceCurFrame=face_recognition.face_locations(imgs)
    encodeCurFrame =face_recognition.face_encodings(imgs,faceCurFrame)

    for encodeFace,faceLoc in zip(encodeCurFrame,faceCurFrame):
        matches=face_recognition.compare_faces(encodeList,encodeFace)
        faceDis = face_recognition.face_distance(encodeList,encodeFace)
        #print(faceDis)
        matchIndex =np.argmin(faceDis)

        if matches[matchIndex]:
            name=classNames[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1=faceLoc
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)

            


    cv2.imshow('webcam',img)
    cv2.waitKey(1)    

            
    


faceLocTest=face_recognition.face_locations(imgTest)[0]
encodeTest =face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)



imgTest =face_recognition.load_image_file('praveen.jpg')
imgTest =cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)
