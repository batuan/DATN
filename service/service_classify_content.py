__author__ = "Trungdq"
__email__ = "trungdq1912@gmail.com"
__version__ = '1.0'

import pickle
import sys
import os
import pandas as pd
from flask import Blueprint, request

from algorithm.data_preprocess_main import FEATURE_EXTRACTION, PREPROCESS

'''main function'''

from goose3 import Goose
from goose3.configuration import Configuration
import json
config = Configuration()
config.strict = False  # turn of strict exception handling
config.browser_user_agent = 'Mozilla 5.0'  # set the browser agent string
config.http_timeout = 5.05  # set http timeout in seconds
with Goose(config) as g:
    pass

def getContent(url):
    try:
        content = g.extract(url)
        rs = {
            "url": url,
            "type": None,
            "title": content.title,
            "description": content.meta_description
        }
    except:
        rs = {
            "url": "",
            "type": None,
            "title": "",
            "description": ""
        }
        print("Can't get content error url")
    return rs

def loadConfig(pathConfig=None):
    config = 'config/conf.json'
    if pathConfig is None:
        with open('config/conf.json') as json_data_file:
            config = json.load(json_data_file)
    else:
        if os.path.exists(pathConfig):
            with open(pathConfig) as json_data_file:
                config = json.load(json_data_file)
        else:
            print('==========> File config not exists <==========')
            sys.exit(0)
    return config

def loadModel(jsonConfig: json):
    path_model = jsonConfig["model"]
    path_vector = jsonConfig["vectorizer"]
    path_selector = jsonConfig["selector"]
    if path_model == None or os.path.exists(path_model) == None:
        print('==========> File model not exists <==========')
        sys.exit(0)
    if path_vector == None or os.path.exists(path_vector) == None:
        print('==========> File vector not exists <==========')
        sys.exit(0)
    if path_selector == None or os.path.exists(path_selector) == None:
        print('==========> File selector not exists <==========')
        sys.exit(0)
    model = pickle.load(open(path_model, 'rb'))
    vector = pickle.load(open(path_vector, 'rb'))
    selector = pickle.load(open(path_selector, 'rb'))
    return model, vector, selector

def predictionURL(model, vector, selector, url):
    arrayJson = []
    if type(url) == str:
        contentJson = getContent(url)
        arrayJson.append(contentJson)
    elif type(url) == list:
        size = len(url)
        for i in range(size):
            contentTmp = getContent(url[i])
            arrayJson.append(contentTmp)
    data = pd.DataFrame(arrayJson)
    features = FEATURE_EXTRACTION(data).extract_properties()
    features_preprocess = PREPROCESS(features, None).processWithTFIDF(vector, selector)
    prediction = model.predict(features_preprocess)
    return prediction


config = loadConfig(pathConfig=None)
model, vector, selector = loadModel(config["content"])

classify = Blueprint('classify', __name__, url_prefix='/classify')

@classify.route('/url-with-content', methods=['POST'])
def classifyURLWithContent():
    data_json = request.json
    array_url = data_json.get("url")
    prediction = predictionURL(model, vector, selector, array_url)
    prediction = prediction.tolist()
    prediction_rs = []
    size = len(array_url)
    for i in range(size):
        tmp = {
            "url": array_url[i],
            "type": prediction[i]
        }
        prediction_rs.append(tmp)
    rs = {
        "payload": prediction_rs,
        "status": 0,
        "description": "Results are returned to you"
    }
    return rs




# if __name__ == '__main__':
#     size_sys = len(sys.argv)
#     config = None
#     if size_sys == 3 and sys.argv[1] == '--config':
#         config = loadConfig(sys.argv[2])
#     else:
#         config = loadConfig(pathConfig=None)
#     print('========>>>> Loading model')
#     model, vector, selector = loadModel(config["content"])
#     print('========>>>> Loading model DONE')
#     predictionURL(model, vector, selector, url=['http://dothi.net/cho-thue-nha-mat-pho-duong-quang-trung-61/cho-thue-nha-mt-quang-trung-q9-dt-12x30m-tret-gac-gia-40trth-pr12637637.htm~'])