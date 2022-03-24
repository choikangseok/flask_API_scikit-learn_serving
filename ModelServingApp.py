# flask_server.py
import tensorflow.keras as keras
import numpy as np
from config import *
from rf_feature_extract import *
from cnn_feature_extract import *
from google_repuCheck import *
from flask import Flask, request
from keras.models import load_model
import pickle


#random forest model
with open(RF_MODEL, 'rb') as rf_file:
    rf_model = pickle.load(rf_file)

#cnn model
with open(CNN_MODEL, 'rb') as cnn_file:
    cnn_model = load_model.load(cnn_file)


app = Flask(__name__)

@app.route('/rfcnn', methods=['POST'])
def rfcnn():
    result = {}

    data = request.get_json()
    rf_feature = np.array(rf_feature_extract(data['URL'])) # Feature_extract() return list
    # rf_feature : [1,2,3,4,5,1 ...]
    cnn_feature = np.array(cnn_feature_extract(data['URL'])) # Feature_extract() return list
    # cnn_feature : [[[255, 255, 255], [255, 255, 255], [255, 255,...
    result['RFmodel'] = {'predict' : int(rf_model.predict([rf_feature])*100)}
    result['CNNmodel'] = {'predict' : int(cnn_model.predict([cnn_feature])*100)}
    result['SafeBrowsing'] = {'malicious' : repuCheck(data['URL'])}





    return result, 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, threaded=True, debug=True)
