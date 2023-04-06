import cv2
import numpy as np
import os 
from PIL import Image


def sort_images(emotion):
    with open(f'{emotion}.txt') as f:
        img = [line.strip() for line in f]

        # traverse through all images and save the anger image in new anger folder
        for image in img:
            loadedImage = cv2.imread("images/"+image)
            cv2.imwrite(f"data_set/{emotion}/"+image,loadedImage)



def rename_images(emotion, id):
    with open(f'{emotion}.txt','r') as f:
        images = [line.strip() for line in f]

        face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        face_id = id

        count = 0

        for image in images:
            img = cv2.imread(f"data_set/{emotion}/"+image)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray,1.3,5)

            for (x,y,w,h) in faces:
                cv2.rectangle(img, (x,y),(x+w,y+h), (255,0,0),2)
                count+=1

                cv2.imwrite("dataset/User."+str(face_id)+"."+str(count)+".jpg", gray[y:y+h,x:x+w])



os.system('cmd /c "git clone https://github.com/misbah4064/facial_expressions.git"')

os.chdir("facial_expressions")
    
os.system('cmd /c "md data_set"')
os.chdir("data_set")
os.system('cmd /c "md anger happy neutral sad"')
os.chdir("..")

 
emotions = ["anger",'happy','neutral','sad']


for i in emotions:
    sort_images(i)


os.system('cmd /c "md dataset"')

for i in range(len(emotions)):
    rename_images(emotions[i], i)


os.system('cmd /c "md trainer"')

