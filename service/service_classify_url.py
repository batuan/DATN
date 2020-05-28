__author__ = "Trungdq"
__email__ = "trungdq1912@gmail.com"
__version__ = '1.0'

import pandas as pd
from flask import Blueprint, request
from algorithm.data_preprocess_main import FEATURE_EXTRACTION, PREPROCESS
from Utility import Utility_func
'''main function'''

config = Utility_func.loadConfig(pathConfig=None)
list_model = []
try:
    list_model = config['list-model']
except:
    print("=============> You need config argument list_model into file config <=============")
    exit(0)

models = {}

for model in list_model:
    try:
        config_model = config[model]
        modelTmp, vectorTmp, selectorTmp = Utility_func.loadModel(config_model)
        models[model] = {
            "model": modelTmp,
            "vectorizer": vectorTmp,
            "selector": selectorTmp,
        }
    except:
        print("==============> Model not exists <==============")

def predictionURL(model, vector, selector, url):
    arrayJson = [];
    if type(url) == str:
        arrayJson.append({"url": url})
    elif type(url) == list:
        for item in url:
            tmp = {
                "url": item
            }
            arrayJson.append(tmp)
    data = pd.DataFrame(arrayJson)
    features = FEATURE_EXTRACTION(data).extract()
    features_preprocess = PREPROCESS(features, None).processWithTFIDF(vector, selector)
    prediction = model.predict(features_preprocess)
    return prediction

classify_url = Blueprint('classify_url', __name__, url_prefix='/classify-url')

@classify_url.route('/url-not-content', methods=['POST'])
def classifyURLNotContent():
    data_json = request.json
    array_url = data_json.get("url")
    domain = data_json.get("domain")
    if domain == None or domain.strip() == "":
        return {
            "payload": [],
            "description": "You lost param domain",
            "status": 1
        }
    model_prop = models[domain]
    prediction = predictionURL(model_prop['model'], model_prop['vectorizer'], model_prop['selector'], array_url)
    prediction = prediction.tolist()
    prediction_rs = []
    size = len(array_url)
    for i in range(size):
        tmp = {
            "url": array_url[i],
            "type": prediction[i]
        }
        prediction_rs.append(tmp)
    return {
        "payload": prediction_rs,
        "status": 0,
        "description": "Results are returned to you"
    }