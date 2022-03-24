# flask_server.py

import numpy as np
import config
import logging
import tensorflow as tf
import rf_feature_extract
import cnn_feature_extract
from flask import Flask, request
import pickle


RF_MODEL = "model/RF.pkl"
CNN_MODEL = "model/CNN.h5"


with open(RF_MODEL, 'rb') as file:
    rf_model = pickle.load(file)

with open(CNN_MODEL, 'rb') as file:
    cnn_model = pickle.load(file)


app = Flask(__name__)
@app.route('/rfcnn', methods=['POST'])
def inference():
    result = {}

    data = request.get_json()
    rf_feature = np.array(rf_feature_extract(data['URLs'])) # Feature_extract() return list
    # rf_feature : [1,2,3,4,5,1 ...]
    cnn_feature = np.array(cnn_feature_extract(data['URLs'])) # Feature_extract() return list
    # cnn_feature : [[[255, 255, 255], [255, 255, 255], [255, 255,...
    result['RFmodel'] = {'predict' : int(rf_model.predict([rf_feature])*100)}
    result['CNNmodel'] = {'predict' : int(cnn_model.predict([cnn_feature])*100)}


    logging.debug(f"[result] {result}")

    return result, 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, threaded=False)
