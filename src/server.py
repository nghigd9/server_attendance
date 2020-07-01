import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the fourth GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

import argparse
import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS
from gevent.pywsgi import WSGIServer
import time
from Recognize import Recognize
import json

# import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.InteractiveSession(config=config)
   


ap = argparse.ArgumentParser()

ap.add_argument("--mymodel", default="outputs/my_model.h5",
    help="Path to recognizer model")
ap.add_argument("--le", default="outputs/le.pickle",
    help="Path to label encoder")
ap.add_argument("--embeddings", default="outputs/embeddings.pickle",
    help='Path to embeddings')
ap.add_argument("--video-out", default="../datasets/videos_output/stream_test.mp4",
    help='Path to output video')


ap.add_argument('--image-size', default='112,112', help='')
ap.add_argument('--model', default='../insightface/models/model-r100-ii/model,0', help='path to load model.')
# ap.add_argument('--model', default='../insightface/models/model-r50-am-lfw/model,0', help='path to load model.')
# ap.add_argument('--model', default='../insightface/models/model-r34-amf/model,0', help='path to load model.')
ap.add_argument('--ga-model', default='', help='path to load model.')
ap.add_argument('--gpu', default=0, type=int, help='gpu id')
ap.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
ap.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
ap.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
ap.add_argument('--model_classify', type=str,default='outputs/my_model.h5')
args = ap.parse_args()

recognize = Recognize(args)

app = Flask(__name__)
CORS(app)
@app.route("/", methods = ['GET'])
def main():
    print("/: " + str(datetime.datetime.now()), flush=True)
    return "server is running on  port : 1602"


@app.route("/recog", methods = ['POST'])
def recog():
    print("")
    print("/recog : ", flush=True)
    try:
        time_start = time.time()
        request_json = request.get_json()
        postFaces = {
            'faces': request_json.get('faces')
        }
        faces = postFaces['faces']
        reses = []
        for face in faces:
            response = recognize.recog_image(face['base64_image'])
            print("response: ", response)
            reses.append((response, face['idface']))
        print(reses)
        time_recog= time.time() - time_start
        print("/recog --- time process:", time_recog)
        return jsonify(reses)
    except Exception as e:
        print(e, flush=True)

@app.route("/reco", methods = ['POST'])
def reco():
    print("")
    print("/reco :" + str(datetime.datetime.now()), flush=True)
    try:
        time_start = time.time()
        request_json = request.get_json()
        # print(request_json)
        postFace = {
            'face': request_json.get('imgbase64')
        }
        face = postFace['face']
        print(face[0:100])
        response = recognize.recog_image(face)
        print(response)
        time_recog= time.time() - time_start
        print("/reco --- time process:", time_recog)
        return jsonify(response)
    except Exception as e:
        print(e, flush=True)

# @app.route("/api/login", method= ['POST'])
# def login():
#     request_json = request.get_json()
if __name__ == "__main__":
    print("Server is running...")
    http_server = WSGIServer(('0.0.0.0', 1602), app)
    # http_server = WSGIServer(('localhost', 1602), app)
    http_server.serve_forever()
    