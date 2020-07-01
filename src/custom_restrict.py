import sys
sys.path.append('../insightface/deploy')
sys.path.append('../insightface/src/common')

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=config)
# import ctypes
# hllDll = ctypes.WinDLL("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.0\\bin\\cudart64_100.dll")

from keras.models import load_model
from mtcnn.mtcnn import MTCNN
from imutils import paths
import face_preprocess
import numpy as np
import face_model
import argparse
import pickle
import time
import dlib
import cv2
import os

ap = argparse.ArgumentParser()

ap.add_argument("--mymodel", default="outputs/my_model.h5",
    help="Path to recognizer model")
ap.add_argument("--le", default="outputs/le.pickle",
    help="Path to label encoder")
ap.add_argument("--embeddings", default="outputs/embeddings.pickle",
    help='Path to embeddings')
ap.add_argument("--video-out", default="../datasets/videos_output/video_test.mp4",
    help='Path to output video')
ap.add_argument("--video-in", default="../datasets/videos_input/GOT_actor.mp4")


ap.add_argument('--image-size', default='112,112', help='')
# ap.add_argument('--model', default='../insightface/models/model-r100-ii/model,0', help='path to load model.')
ap.add_argument('--model', default='../insightface/models/model-r50-am-lfw/model,0', help='path to load model.')
# ap.add_argument('--model', default='../insightface/models/model-r34-amf/model,0', help='path to load model.')
ap.add_argument('--ga-model', default='', help='path to load model.')
ap.add_argument('--gpu', default=0, type=int, help='gpu id')
ap.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
ap.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
ap.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')

args = ap.parse_args()

# Load embeddings and labels
data = pickle.loads(open(args.embeddings, "rb").read())
le = pickle.loads(open(args.le, "rb").read())

embeddings = np.array(data['embeddings'])
labels = le.fit_transform(data['names'])

# Initialize detector
detector = MTCNN()

# Initialize faces embedding model
embedding_model =face_model.FaceModel(args)

# Load the classifier model
model = load_model('outputs/my_model.h5')

# Define distance function
def findCosineDistance(vector1, vector2):
    """
    Calculate cosine distance between two vector
    """
    vec1 = vector1.flatten()
    vec2 = vector2.flatten()

    a = np.dot(vec1.T, vec2)
    b = np.dot(vec1.T, vec1)
    c = np.dot(vec2.T, vec2)
    return 1 - (a/(np.sqrt(b)*np.sqrt(c)))

def CosineSimilarity(test_vec, source_vecs):
    """
    Verify the similarity of one vector to group vectors of one class
    """
    cos_dist = 0
    for source_vec in source_vecs:
        cos_dist += findCosineDistance(test_vec, source_vec)
    return cos_dist/len(source_vecs)

# Initialize some useful arguments
cosine_threshold = 0.8
proba_threshold = 0.85
comparing_num = 5
trackers = []
texts = []
frames = 0

# Start streaming and recording
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
save_width = 600
save_height = int(600/frame_width*frame_height)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    start_time = time.time()
    ret, frame = cap.read()
    frames += 1
    frame = cv2.resize(frame, (save_width, save_height))
    orgin_frame = frame
    cv2.rectangle(orgin_frame, (122, 87), (450, 350), (0,255,0), 2)
    frame = frame[87:350, 122:450]
    if frames % 3 != 0:
        detect_time = time.time()
        bboxes = detector.detect_faces(frame)
        print("detect faces time:", str(time.time() -  detect_time))
        if len(bboxes) != 0:
            for bboxe in bboxes:
                bbox = bboxe['box']
                bbox = np.array([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])
                landmarks = bboxe['keypoints']
                landmarks = np.array([landmarks["left_eye"][0], landmarks["right_eye"][0], landmarks["nose"][0], landmarks["mouth_left"][0], landmarks["mouth_right"][0],
                                        landmarks["left_eye"][1], landmarks["right_eye"][1], landmarks["nose"][1], landmarks["mouth_left"][1], landmarks["mouth_right"][1]])
                landmarks = landmarks.reshape((2,5)).T
                nimg = face_preprocess.preprocess(frame, bbox, landmarks, image_size='112,112')
                nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
                nimg = np.transpose(nimg, (2,0,1))
                emb_time = time.time()
                embedding = embedding_model.get_feature(nimg).reshape(1,-1)
                print("extract feature time: " + str(time.time() - emb_time))

                text = "Unknown"

                # Predict class
                preds = model.predict(embedding)
                preds = preds.flatten()
                # Get the highest accuracy embedded vector
                j = np.argmax(preds)
                proba = preds[j]
                # Compare this vector to source class vectors to verify it is actual belong to this class
                match_class_idx = (labels == j)
                match_class_idx = np.where(match_class_idx)[0]
                selected_idx = np.random.choice(match_class_idx, comparing_num)
                compare_embeddings = embeddings[selected_idx]
                # Calculate cosine similarity
                cos_similarity = CosineSimilarity(embedding, compare_embeddings)
                if cos_similarity < cosine_threshold and proba > proba_threshold:
                    name = le.classes_[j]
                    text = "{}".format(name)
                    print("Recognized: {} <{:.2f}> ".format(name, proba*100))
                y = bbox[1] - 10 if bbox[1] - 10 > 10 else bbox[1] + 10
                cv2.putText(frame, text, (bbox[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255,0,0), 2)
    # orgin_frame = cv2.resize(frame, (1280, 720), cv2.INTER_AREA)
    cv2.imshow("Frame", orgin_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    print("1 loop time:", time.time() - start_time)

cap.release()
cv2.destroyAllWindows()
