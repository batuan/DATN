__author__ = "Trungdq"
__email__ = "trungdq1912@gmail.com"
__version__ = '1.0'

import pickle
from sklearn.model_selection import train_test_split
from algorithm.data_preprocess_main import FETCH_DATA, FEATURE_EXTRACTION, PREPROCESS
from Utility import Utility_func
import sys
import os

from algorithm.predictive_model import Classify

'''main function'''


if __name__ == '__main__':
    size_sys = len(sys.argv)
    mapConfig = {
        '--config': None,
        '--name': None,
        '--train-model': None,
        '--test-size': 0.000001,
        '--predict': None
    }
    if (size_sys - 1) != 0 and ((size_sys - 1) % 2) == 0:
        for i in range(size_sys):
            if i != 0 and (i % 2) != 0:
                mapConfig[sys.argv[i]] = sys.argv[i + 1]
    else:
        print("=======> You lost arguments <=============")
        exit(0)

    config = Utility_func.loadConfig(mapConfig['--config'])

    if mapConfig['--name'] == None:
        print("=======> You lost arguments <=============")
        exit(0)
    else:
        nameModel = mapConfig['--name'];
        # check config
        configModel = None
        try:
            configModel = config[nameModel]
        except:
            print("=======> Haven't model into file config <=============")
            exit(0)
        if mapConfig['--train-model'] != None and mapConfig['--train-model'].lower() == 'true':
            pathData = configModel['data-path']
            dataSet = configModel['data-train']
            modelPath = configModel['model']
            vectorPath = configModel['vectorizer']
            selectorPath = configModel['selector']
            directory = os.path.dirname(modelPath)
            if not os.path.exists(directory):
                os.makedirs(directory,  exist_ok=True)
            data, label = FETCH_DATA(pathData, dataSet).fetch_url()
            features = FEATURE_EXTRACTION(data).extract()
            test_size_param = float(mapConfig['--test-size'])
            features_train, features_test, label_train, label_test = train_test_split(features, label, test_size=test_size_param)
            features_train_preprocess, vectorizer, selector = PREPROCESS(features_train, label_train).process()
            # save vectorizer, selector
            if not pathData.endswith('/'):
                pathData += '/'
            pickle.dump(vectorizer, open(vectorPath, 'wb'))
            pickle.dump(selector, open(selectorPath, 'wb'))
            '''Return model result'''
            '''
                Binomial Model Output:
                NB(features_train_BNB, label_train)
            '''
            model = Classify(features_train_preprocess, label_train).Support_Vector()
            pickle.dump(model, open(modelPath, 'wb'))
        elif mapConfig['--predict'] != None:
            str_url = mapConfig['--predict']
            list_url = str_url.split(",")
            data = []
            for item in list_url:
                data.append(item.strip())
            model = pickle.load(open(configModel['model'], 'rb'))
            vector = pickle.load(open(configModel['vectorizer'], 'rb'))
            selector = pickle.load(open(configModel['selector'], 'rb'))
            features = FEATURE_EXTRACTION().extract_url(data)
            features_preprocess_test = PREPROCESS(features, None).processWithTFIDF(vector, selector)
            predict = model.predict(features_preprocess_test)
            print("========> Resutl: " + str(predict))
        else:
            print("===========> You can select option --train-model or --predict")
