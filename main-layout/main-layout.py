__author__ = "Trungdq"
__email__ = "trungdq1912@gmail.com"
__version__ = '1.0'

import json
import pandas as pd
import numpy as np
import os
from time import time
from sklearn.model_selection import train_test_split
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
import pickle
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import KFold
from sklearn.svm import SVC


def fetch_data(data_frame, array):
    data = data_frame
    try:
        print("==========Preprocessing the data==========")
        print("Start labelling data ...")
        lab = set(data['label'].values)
        lab = dict(enumerate(lab, 1))
        lab = dict(zip(lab.values(), lab.keys()))

        '''convert keys to values and values to keys.
                This helps to turn the label into numerics.
                for classification'''

        label = list(map(lab.get, list(data['label'].values)))
        data['label'] = pd.Series(label).values
        data = data.loc[:, array]
        print('DONE LOADING DATA')
        print(20 * '*')
        return data, label
    except Exception as e:
        print(e)
        pass
    finally:
        print('finnished.')


def train_data_castboost():
    df = pd.read_json('../data/data-main-layout/rs-content-1.json')
    df = df.drop(columns=['xpath', 'content', 'imgSize', 'imgNum'])
    # print(df)
    train_features = ['fontSizeAbsolute', 'linkNumAbsolute', 'interactionSize',
                      'innerTextLength', 'innerHTMLLength', 'fontSize', 'blockCenterX', 'blockCenterY',
                      'blockRectHeight', 'blockRectWidth', 'jaccard', 'contact',
                      'architecture', 'address', 'suface', 'time', 'price', 'iterator', 'order']

    data, label = fetch_data(df, train_features)
    features_train, features_test, label_train, label_test = train_test_split(data, label, test_size=0.15)
    # text feature
    # text_feature = [5, 10, 11,12, 13, 14, 15, 16, 17]
    # tokenizers
    # tokenizers = [{
    #     'tokenizer_id': 'Space',
    #     'delimiter': ' ',
    #     'separator_type': 'ByDelimiter',
    # }]
    # dictionaries
    # dictionaries = [{
    #     'dictionary_id': 'Unigram',
    #     'max_dictionary_size': '500000',
    #     'token_level_type': 'Word',
    #     'gram_count': '1',
    #     'tokenizerId': 'Space'
    #
    # }]
    # Initialize CatBoostClassifier
    model = CatBoostClassifier(iterations=400, learning_rate=0.015, model_shrink_rate=0.92,
                               depth=12, loss_function='Logloss', random_seed=99)

    cat_features = ['jaccard', 'contact',
                    'architecture', 'address', 'suface', 'time', 'price', 'iterator']
    # # Fit model
    train_pool = Pool(features_train, label_train, cat_features=cat_features)
    test_pool = Pool(features_test, label_test, cat_features=cat_features)

    model.fit(train_pool, eval_set=test_pool)

    for score, name in sorted(zip(model.feature_importances_, model.feature_names_), reverse=True):
        print('{}\t{}'.format(name, score))

    predicted = model.predict(features_test)
    report = classification_report(label_test, predicted)
    print('==== CLASSIFICATION REPORT ======')
    print(report)
    print('*' * 50)
    print('==== CONFUSION MATRIX ======')
    print(confusion_matrix(label_test, predicted, labels=[1, 2]))
    print('*' * 50)

    return model


def train_data_catboost_kfold():
    df = pd.read_json('../data/data-main-layout/rs-content.json')
    df = df.drop(columns=['xpath', 'content'])
    # print(df)
    train_features = ['fontSizeAbsolute', 'linkNumAbsolute', 'interactionSize',
                      'innerTextLength', 'innerHTMLLength', 'imgSize',
                      'fontSize', 'imgNum', 'blockCenterX', 'blockCenterY',
                      'blockRectHeight', 'blockRectWidth', 'fontWeight']
    data, label = fetch_data(df, train_features)

    features_train, features_test, label_train, label_test = train_test_split(data, label, test_size=0.2)
    kf = KFold(n_splits=5)
    Y = np.asarray(label_train)
    final_accuracy = []
    interator_KF = enumerate(kf.split(features_train))
    for FOLD_NO, (train_index, test_index) in interator_KF:
        print('Fold: {}'.format(FOLD_NO))
        X_train = features_train.iloc[train_index]
        X_test = features_train.iloc[test_index]
        Y_train = Y[train_index]
        Y_test = Y[test_index]

        # Initialize CatBoostClassifier
        model = CatBoostClassifier(iterations=1000, learning_rate=0.022, model_shrink_rate=0.95,
                                   depth=10, loss_function='Logloss', random_seed=99, task_type='GPU',
                                   cat_features=cat_features)

        cat_features = []
        # Fit model
        train_pool = Pool(X_train, Y_train, cat_features=cat_features)
        test_pool = Pool(X_test, Y_test, cat_features=cat_features)

        model.fit(train_pool, eval_set=test_pool)
        accuracy = model.score(X_train, Y_train)
        final_accuracy.append(accuracy)
        for score, name in sorted(zip(model.feature_importances_, model.feature_names_), reverse=True):
            print('{}\t{}'.format(name, score))

        predicted = model.predict(X_test)
        report = classification_report(Y_test, predicted)
        print('==== CLASSIFICATION REPORT ======')
        print(report)
        print('*' * 50)
        print('==== CONFUSION MATRIX ======')
        print(confusion_matrix(Y_test, predicted, labels=[1, 2]))
        print('*' * 50)
        print('==== PRECISION RECALL FSCOR SUPPORT WEIGHTED======')
        '''Note that this is same as classification report
        but without Support. And it is the weighted average'''
        print(precision_recall_fscore_support(Y_test, predicted, average='weighted'))
        print('*' * 50)

    print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(final_accuracy), np.std(final_accuracy)))
    print('*' * 50)
    print('====================== CatBoost Machine 100% Completed ===========================\n\n')
    print('====================== Prediction for TEST =======================================\n\n')
    prediction = model.predict(features_test)
    report = classification_report(Y_test, predicted)
    print('==== CLASSIFICATION REPORT ======')
    print(report)
    print('*' * 50)
    print('==== CONFUSION MATRIX ======')
    print(confusion_matrix(Y_test, predicted, labels=[1, 2]))
    print('*' * 50)
    return model


def train_data_svm():
    df = pd.read_json('../data/data-main-layout/rs-content.json')
    df = df.drop(columns=['xpath', 'content'])

    train_features = ['fontSizeAbsolute', 'linkNumAbsolute', 'interactionSize',
                      'innerTextLength', 'innerHTMLLength',
                      'fontSize', 'blockCenterX', 'blockCenterY',
                      'blockRectHeight', 'blockRectWidth', 'fontWeight']

    data, label = fetch_data(df, train_features)
    features_train, features_test, label_train, label_test = train_test_split(data, label, test_size=0.15)
    kf = KFold(n_splits=5)
    Y = np.asarray(label_train)
    final_accuracy = []
    interator_KF = enumerate(kf.split(features_train))
    for FOLD_NO, (train_index, test_index) in interator_KF:
        print('Fold: {}'.format(FOLD_NO))
        X_train = features_train.iloc[train_index].to_numpy()
        X_test = features_train.iloc[test_index].to_numpy()
        Y_train = Y[train_index]
        Y_test = Y[test_index]
        clf = SVC(kernel='poly', C=1.0)
        clf.fit(X_train, Y_train)
        start_time = time()
        accuracy = clf.score(X_train, Y_train)
        final_accuracy.append(accuracy)
        print("training time:", round(time() - start_time, 2), "secs")
        start_time2 = time()
        prediction = clf.predict(X_test)
        print("predict time:", round(time() - start_time2, 2), "secs")
        target_names = ['1', '2']

        print('==== CLASSIFICATION REPORT ======')
        print(classification_report(Y_test, prediction, target_names=target_names))
        print('*' * 50)
        print('==== CONFUSION MATRIX ======')
        print(confusion_matrix(Y_test, prediction, labels=[1, 2]))
        print('*' * 50)
        print('==== PRECISION RECALL FSCOR SUPPORT WEIGHTED======')
        '''Note that this is same as classification report
        but without Support. And it is the weighted average'''
        print(precision_recall_fscore_support(Y_test, prediction, average='weighted'))
        print('*' * 50)

    print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(final_accuracy), np.std(final_accuracy)))
    print('*' * 50)
    print('====================== 100% Support Vector Machine 100% Completed ===========================\n\n')

    return clf


if __name__ == '__main__':
    os.chdir(os.getcwd())
    with open('../data/data-main-layout/rs-1.json') as json_file:
        data = json.load(json_file)

    model = train_data_svm()

    print("******************************* TEST ****************************************")
    df = pd.read_json('../data/data-main-layout/result-2.json')
    df = df.drop(columns=['xpath'])
    train_features = ['fontSizeAbsolute', 'linkNumAbsolute', 'interactionSize',
                      'innerTextLength', 'innerHTMLLength', 'fontSize', 'blockCenterX', 'blockCenterY',
                      'blockRectHeight', 'blockRectWidth']
    data, label = fetch_data(df, train_features)
    Y = np.asarray(label)
    predicted = model.predict(data)
    report = classification_report(Y, predicted)
    print('==== CLASSIFICATION REPORT ======')
    print(report)
    print('*' * 50)
    print('==== CONFUSION MATRIX ======')
    print(confusion_matrix(Y, predicted, labels=[1, 2]))
    print('*' * 50)
    size = len(predicted)
    for it in range(size):
        if (predicted[it] != label[it]):
            print(it)
            print(data.iloc[it])
