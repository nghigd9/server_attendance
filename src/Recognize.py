import base64
import io
import sys
import uuid

from PIL import Image

sys.path.append('../insightface/deploy')
sys.path.append('../insightface/src/common')

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

# import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         # Restrict TensorFlow to only use the fourth GPU
#         tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)


# # Load embeddings and labels
# data = pickle.loads(open(args.embeddings, "rb").read())
# le = pickle.loads(open(args.le, "rb").read())
#
# embeddings = np.array(data['embeddings'])
# labels = le.fit_transform(data['names'])
#
# # Initialize detector
# detector = MTCNN()
#
# # Initialize faces embedding model
# embedding_model =face_model.FaceModel(args)
#
# # Load the classifier model
# model = load_model('outputs/my_model.h5')

class Recognize:
    def __init__(self, args):
        self.le = args.le
        self.data = pickle.loads(open(args.embeddings, "rb").read())
        self.embeddings = np.array(self.data['embeddings'])
        self.labelEncode = pickle.loads(open(self.le, "rb").read())
        self.labels = self.labelEncode.fit_transform(self.data['names'])
        self.detector = MTCNN()
        self.embedding_model = face_model.FaceModel(args)
        self.classifying_model = load_model(args.model_classify)

    def findCosineDistance(self, vector1, vector2):
        """
        Calculate cosine distance between two vector
        """
        vec1 = vector1.flatten()
        vec2 = vector2.flatten()

        a = np.dot(vec1.T, vec2)
        b = np.dot(vec1.T, vec1)
        c = np.dot(vec2.T, vec2)
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

    def CosineSimilarity(self, test_vec, source_vecs):
        """
        Verify the similarity of one vector to group vectors of one class
        """
        cos_dist = 0
        for source_vec in source_vecs:
            cos_dist += self.findCosineDistance(test_vec, source_vec)
        return cos_dist / len(source_vecs)

    def saveImage(self, np_image):
        np_image = np_image[...,::-1]
        save_image = Image.fromarray(np_image)
        save_image.save('../datasets/log/' + str(uuid.uuid4()) + '.jpg' )
        print('saved image to dataset')

    def recog_image(self, base64_image):
        cosine_threshold = 0.8
        proba_threshold = 0.85
        comparing_num = 5
        ####
        result = []
        try:
            file_like = io.BytesIO(base64.b64decode(base64_image))
            image = Image.open(file_like)
            np_image = np.array(image)
            print(np_image.shape)
            np_image = cv2.cvtColor(np.array(np_image), cv2.COLOR_BGR2RGB)
            print(np_image.shape)
            self.saveImage(np_image)
            # np_image.save('../datasets/log/' + str(uuid.uuid4()) +'.jpg')
            bboxes = self.detector.detect_faces(np_image)
            if len(bboxes) != 0:
                for bboxe in bboxes:
                    bbox = bboxe['box']
                    bbox = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                    landmarks = bboxe['keypoints']
                    landmarks = np.array([landmarks["left_eye"][0], landmarks["right_eye"][0], landmarks["nose"][0],
                                          landmarks["mouth_left"][0], landmarks["mouth_right"][0],
                                          landmarks["left_eye"][1], landmarks["right_eye"][1], landmarks["nose"][1],
                                          landmarks["mouth_left"][1], landmarks["mouth_right"][1]])
                    landmarks = landmarks.reshape((2, 5)).T
                    nimg = face_preprocess.preprocess(np_image, bbox, landmarks, image_size='112,112')
                    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
                    nimg = np.transpose(nimg, (2, 0, 1))
                    embedding = self.embedding_model.get_feature(nimg).reshape(1, -1)
                    # print(embedding)
                    text = "Unknown"

                    # Predict class
                    preds = self.classifying_model.predict(embedding)
                    preds = preds.flatten()
                    # Get the highest accuracy embedded vector
                    j = np.argmax(preds)
                    proba = preds[j]
                    # Compare this vector to source class vectors to verify it is actual belong to this class
                    match_class_idx = (self.labels == j)
                    match_class_idx = np.where(match_class_idx)[0]
                    selected_idx = np.random.choice(match_class_idx, comparing_num)
                    compare_embeddings = self.embeddings[selected_idx]
                    # Calculate cosine similarity
                    cos_similarity = self.CosineSimilarity(embedding, compare_embeddings)
                    if cos_similarity < cosine_threshold and proba > proba_threshold:
                        name = self.labelEncode.classes_[j]
                        text = "{}".format(name)
                        print("<Recognize> :  {} <{:.2f}>".format(name, proba * 100))

                    y = bbox[1] - 10 if bbox[1] - 10 > 10 else bbox[1] + 10
                    # cv2.putText(np_image, text, (bbox[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                    # cv2.rectangle(np_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                    top_left_x, top_left_y, bottom_right_x, bottom_right_y = bbox
                    result.append({"maSv": text, 
                        "top_left_x": int(top_left_x),
                        "top_left_y": int(top_left_y),
                        "bottom_right_x": int(bottom_right_x),
                        "bottom_right_y": int(bottom_right_y)
                        })
            return result
        except Exception as e:
            print("sadddddddddddd:", e)
            return result
        # cv2.imshow("Figure", np_image)
        # cv2.waitKey(0)
        # cv2.imwrite(args.image_out, np_image)
        # cv2.destroyAllWindows()
