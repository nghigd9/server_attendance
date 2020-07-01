# import the necessary packages
# from imutils.video import VideoStream
# from imutils.video import FPS
import argparse
import imutils
import time
import cv2
import threading
import requests
import json, io
import base64
import uuid
from PIL import Image

URL_RECOG_SERVER = "http://localhost:1602/recog"

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)


lock = threading.Lock()

# retur = False

class MyData():
    def __init__(self):
        super().__init__()
        self.postFaces = None
        self.response = None

def video_processs(data):
    
    trackers = []
    ids = []
    cur_face_num = -1
    postData = []

    while(cap.isOpened()):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_temp = face_cascade.detectMultiScale(gray, 1.2, 4)
        faces = []
        for (x, y, w ,h) in faces_temp:
            if w < 100 or h < 125:
                continue
            faces.append( (x, y, w, h))
        if len(faces) != cur_face_num:
            trackers = []
            ids = []
            cur_face_num = len(faces)
            postData = []

            for (x, y, w, h) in faces:
                bbox = (x, y , w, h)
                tracker = cv2.TrackerKCF_create()
                tracker.init(frame, bbox)
                trackers.append(tracker)
                id_num = uuid.uuid4()
                ids.append((id_num , ""))

                face = frame[y:y+h, x:x+w]
                pil_img = Image.fromarray(face)
                buff = io.BytesIO()
                pil_img.save(buff, format="JPEG")
                new_image_string = base64.b64encode(buff.getvalue()).decode("utf-8")
                uid = str(id_num)
                params = {
                    "base64_image": new_image_string,
                    "idface": uid
                }
                postData.append(params)
        if lock.locked() == False:
            data.postFaces = {
                "faces" : postData
            }
        # print("len trackers:", len(trackers))
        for (tracker1, idAndName) in zip(trackers,ids):
            ok, bbox = tracker1.update(frame)
            (x, y, w, h) = bbox
            idn, name = idAndName
            # print("idn:", idn)
            # print("name:", name)
            if ok:
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
                if lock.locked() == False:
                    try:
                        # print("here.")
                        response = data.response
                        jData = json.loads(response.content.decode('utf-8'))
                        print("---------------------------")
                        print(len(jData))
                        for i, (res, faceId) in enumerate(jData):
                            if (faceId == str(idn)):
                                name = res[0]['label']
                    except Exception as es:
                        pass
                cv2.putText(frame, name, (int(x), int(y - 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255),2)
                ids.remove(idAndName)
                ids.append((idn, name))
            else :
                # print("remove track id: ", idn)
                print("remo")
                print(len(trackers))
                cur_face_num = -1
                trackers.remove(tracker)
                ids.remove(idAndName)
        
        cv2.imshow("frame", frame)
        k = cv2.waitKey(1) & 0xff
        if k == ord("q"):
            retur = True
            break


def send(data):
    # counter = 5
    stop = False
    while stop == False:
        lock.acquire()
        try:
            faces = data.postFaces
            start = time.time()
            response = requests.post(URL_RECOG_SERVER, json=faces)
            print("time req :", time.time() - start)
            data.response = response
            jData = json.loads(response.content.decode('utf-8'))
            for (res, faceId) in jData:
                print("res", res[0]['label'])
                print(faceId)
        except Exception  as e:
            print(e)
        lock.release()
        time.sleep(0.1)
        # k = cv2.waitKey(1) & 0xff
        # if k == ord("q"):
        #     retur = True
        #     break
        # counter -= 1

if __name__ == "__main__":
    data = MyData()
    t1 = threading.Thread(target=video_processs, args=(data,))
    t1.start()
    t2 =  threading.Thread(target=send, args=(data,))
    t2.start()
    