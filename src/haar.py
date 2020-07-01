import cv2
import imutils
import time
import base64
from PIL import Image
import numpy as np
import io
import json
import os, requests
import dlib


import threading


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

URL_RECOG_SERVER = "http://localhost:1602/recog"

lock = threading.Lock()


class MyData:
    def __init__(self):
        self.response = None
        self.params = None
        self.name = []

trackers = []
texts = []
frames = 0


def start_recog(data):
    cap = cv2.VideoCapture(0)
    while(cap.isOpened()):
        ret, frame = cap.read()
        start_time = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 4)
        #continue if face too small
        for (x, y, w, h) in faces:
            if w < 100 or h < 100:
                continue
            face = frame[y:y+h, x:x+w]
            pil_img = Image.fromarray(face)
            buff = io.BytesIO()
            pil_img.save(buff, format="JPEG")
            new_image_string = base64.b64encode(buff.getvalue()).decode("utf-8")
            params = {
                "base64_image": new_image_string
            }
            if (lock.locked() == False):
                # print("duoc quyen truy cap.")
                try:
                    data.params = params
                    # print("Main thread:", data.name)
                    cv2.putText(frame, data.name[-1], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                except Exception as e:
                    abs(1)  
        # else:
        #     for tracker, text in zip(trackers,texts):
        #     pos = tracker.get_position()

        #     # unpack the position object
        #     startX = int(pos.left())
        #     startY = int(pos.top())
        #     endX = int(pos.right())
        #     endY = int(pos.bottom())

        #     cv2.rectangle(frame, (startX, startY), (endX, endY), (255,0,0), 2)
        #     cv2.putText(frame, text, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255),2)
        cv2.imshow("camera", frame)
        # print("time 1 for: ", time.time() - start_time)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
def send(data):
    counter = 5
    while True:
        lock.acquire()
        try:
            params = data.params
            start = time.time()
            response = requests.post(URL_RECOG_SERVER, json=params)
            print("time req :", time.time() - start)
            data.response = response
            jData = json.loads(response.content.decode('utf-8'))
            for d in jData:
                print(d['label'])
                data.name.append(d['label'])
        except Exception  as e:
            print(e)
        lock.release()
        time.sleep(1)
        counter -= 1

if __name__ =="__main__":
    data = MyData()
    t1 = threading.Thread(target=start_recog, args=(data,))
    t1.start()
    t2 =  threading.Thread(target=send, args=(data,))
    t2.start()