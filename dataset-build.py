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
os.system('cmd /c "md anger happy neutral sad surprise"')
os.chdir("..")

 
emotions = ["anger",'happy','neutral','sad','surprise']


for i in emotions:
    sort_images(i)


os.system('cmd /c "md dataset"')

for i in range(len(emotions)):
    rename_images(emotions[i], i)


os.system('cmd /c "md trainer"')


path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

# function to get the images and label data
def getImagesAndLabels(path):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids

print ("\n [INFO] Training faces....")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yml
recognizer.write('trainer/trainer.yml') 

# Print the numer of Emotions trained and end program
print("\n [INFO] {0} Emotions trained. Exiting Program".format(len(np.unique(ids))))
