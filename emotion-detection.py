import cv2
import numpy as np
import os 
import requests
import json

def last_fm(payload):
    API_KEY = "f1398a444fa5bd0e7af7c595efa3307c"
    headers = {'user-agent': 'chronic'}
    url = 'https://ws.audioscrobbler.com/2.0/'


    payload['api_key'] = API_KEY
    payload['format'] = 'json'
    payload['limit'] = 1

    response = requests.get(url, headers=headers, params=payload)
    json_data = json.dumps(response.json())
    json_data = json.loads(json_data)
    name = json_data["results"]["albummatches"]['album'][0]['artist']
    url = json_data["results"]["albummatches"]['album'][0]['url']
    return name, url


    
os.chdir("facial_expressions")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
id = 0

# Emotions related to ids: example ==> Anger: id=0,  etc
names = ['Anger', 'Happy', 'Neutral', 'Sad', 'None'] 


id = "unknown"
# cam = cv2.VideoCapture(0)
# # Initialize and start realtime video capture
# cam.set(3, 640) # set video widht
# cam.set(4, 480) # set video height

# # Define min window size to be recognized as a face
# minW = 0.1*cam.get(3)
# minH = 0.1*cam.get(4)
emotion_image = []
emotion_detected = ""

confidence = 0

# while confidence<30:
    # ret, img =cam.read()
    # print("XX")
    # cv2.imshow('img', img)
img = cv2.imread("live_03.jpg")

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale( 
    gray,
    scaleFactor = 1.2,
    minNeighbors = 5,
    # minSize = (int(minW), int(minH)),
    )

for(x,y,w,h) in faces:

    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

    id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
    print(confidence)
    # Check if confidence is less then 100 ==> "0" is perfect match 
    if (confidence > 30 and confidence < 100):
        id = names[id]
        confidence = "  {0}".format(round(100 - confidence))
        emotion_image = img
        print(str(id))
        emotion_detected = str(id)
        
    else:
        id = "unknown"
        confidence = "  {0}%".format(round(100 - confidence))
    
    cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
    cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
        # else:
        #     print("unidentified")


cv2.imwrite("new.jpg",emotion_image) 
print("\n [INFO] Done detecting and Image is saved")
# cam.release()
cv2.destroyAllWindows()

name,url = last_fm( {'method': 'album.search','album':emotion_detected})
print(f"Artist Name:{name}\nURL:{url}")



