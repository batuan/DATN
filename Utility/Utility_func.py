import pickle
import sys
import os
import json

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