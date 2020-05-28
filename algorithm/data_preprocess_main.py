__author__ = "Trungdq"
__email__ = "trungdq1912@gmail.com"
__version__ = '1.0'

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
import pickle

"""
This script Labels the data and extracts features and labels for further processing
The features are extrcted to a list and saved as feature.pkl and labels are saved as label.pkl
"""

class FETCH_DATA(object):

    def __init__(self, path, dataset):
        '''Arhuments:
           path:
               define the path containing the dataset
           dataset:
               the URI datasets we want to work on

       '''
        self.path = path
        self.dataset = dataset

    def fetch_data(self):
        pd.set_option('display.max_columns', None)
        if os.path.exists(self.path):
            try:
                print('=============Preprocessing the data=======================')
                data = pd.read_json(self.path + self.dataset)
                pd.set_option("display.max_columns", None)
                print('DONE LOADING DATA')
                print(20*'*')
                print('Start labelling data ......')
                '''Labelling the dataset'''
                lab = set(data['type'].values)
                lab = dict(enumerate(lab, 1))
                lab = dict(zip(lab.values(), lab.keys()))

                '''convert keys to values and values to keys.
                This helps to turn the label into numerics.
                for classification'''

                label = list(map(lab.get, list(data['type'].values)))
                data['type'] = pd.Series(label).values
                data = data.loc[:, ['url', 'title', 'description', 'type']]
                print('DONE LOADING DATA')
                print(20*'*')
                return data, label
            except Exception as e:
                print(e)
                pass
            finally:
                print('finnished.')

    def fetch_url(self):
        if os.path.exists(self.path):
            try:
                print('=============Preprocessing the data=======================')
                # path = "../../StandaraData/"
                # dataset = "sample.txt"
                # print(self.path + self.dataset)
                data = pd.read_json(self.path + self.dataset)
                pd.set_option("display.max_columns", None)
                print('DONE LOADING DATA')
                print(20*'*')
                print('Start labelling data ......')
                '''Labelling the dataset'''
                lab = set(data['prediction'].values)
                lab = dict(enumerate(lab, 1))
                lab = dict(zip(lab.values(), lab.keys()))

                '''convert keys to values and values to keys.
                This helps to turn the label into numerics.
                for classification'''

                label = list(map(lab.get, list(data['prediction'].values)))
                data['prediction'] = pd.Series(label).values
                data = data.loc[:, ['url', 'prediction']]
                print('DONE LOADING DATA')
                print(20*'*')
                return data, label
            except Exception as e:
                pass
            finally:
                print('finnished.')

    def fetch(self):
        if os.path.exists(self.path):
            try:
                print('=============Preprocessing the data=======================')
                # path = "../../StandaraData/"
                # dataset = "sample.txt"
                data = pd.read_csv(self.path + self.dataset, sep='\t', header=None, error_bad_lines=False)
                data.columns = ["url", "label"]
                print('DONE LOADING DATA')
                print(20*'*')
                print('Start labelling data ......')
                '''Labelling the dataset'''
                lab = set(data['label'].values)
                lab = dict(enumerate(lab, 1))
                lab = dict(zip(lab.values(), lab.keys()))

                '''convert keys to values and values to keys.
                This helps to turn the label into numerics.
                for classification'''

                label = list(map(lab.get, list(data['label'].values)))
                data['label'] = pd.Series(label).values
                data = data.loc[:, ['url', 'label']]
                print('DONE LOADING DATA')
                print(20*'*')
                return data, label
            except Exception as e:
                pass
            finally:
                print('finnished.')


# %% FEATURE_EXTRACTION

class FEATURE_EXTRACTION(object):

    def __init__(self, data=None):
        '''
        Argument:
            data: processed returned by the fetch_data class and
            extract the features as predictors
        '''
        self.data = data

    def convertURL(self, url):
        if (url.startswith('http')):
            splits = url.split('/')
            splits[0] = ""
            splits[1] = ""
            splits[2] = ""
            string = '/'.join(splits)
            return string.replace('//', '')
        return url

    def standraString(self, string):
        #  lowerCase string
        # lowerCase = str(string).lower().strip()
        lowerCase = str(string).strip()
        # replace [\\.,()/:] to space
        replace = re.sub(r'[🚀📌💸\]\[\{\}!@_✓=;\?“|–\\\\\%\\",:.()+/*-]', r' ', lowerCase)
        #replace = re.sub(, r' ', replace)
        # bỏ khoảng trắng giữa các số
        replace = re.sub(r'(?:\s)(?=[0-9])', r'', replace)
        # bỏ các khoảng trắng gần nhau
        replace = re.sub(r'\n', r' ', replace)
        replace = re.sub(r'\s+', r' ', replace)
        return replace

    def extract_feature_content(self, content):
        #standaraStr = self.standraString(content);
        feature = []
        check = False
        pattern_phone = re.compile('(09[0-9]|01[2|6|8|9]|03[2-9]|07[0|9|7|6|8]|08[3|4|5|1|2]|05[6|8|9])+([0-9]{7})')
        isPhone = pattern_phone.search(content)
        if isPhone:
            check = True
            feature.append("phoneContentFeature")
        pattern_suface = re.compile('(dt)(\s*)([0-9]+)|(diện tích)(\s*)([0-9]+)|(dtsd)(\s*)([0-9]+)|(diện tích sử dụng)(\s*)([0-9]+)|'
                                    '([1-9]+\s*m2)|([1-9]+\s*(m|m2)*\s*x\s*[1-9]+(m2|t|m)*)')
        isSuface = pattern_suface.search(content)
        if isSuface:
            check = True
            feature.append('dienTichContentFeature')
        pattern_phongngu = re.compile('([1-9]+\s*phòng\s*ngủ)|([1-9]+\s*pn)|([1-9]+\s*phong\s*ngu)')
        isPhongNgu = pattern_phongngu.search(content)
        if isPhongNgu:
            check = True
            feature.append('phongNguContentFeature')
        pattern_tang = re.compile('([1-9]*\s*(tầng|tang|lầu|lau|trệt|tret))')
        isTang = pattern_tang.search(content)
        if isTang:
            check = True
            feature.append('tangContentFeature')
        return feature, check



    def extract_feature_url(self, url):
        feature = []
        check = False
        pattern_phone = re.compile('(09[0-9]|01[2|6|8|9]|03[2-9]|07[0|9|7|6|8]|08[3|4|5|1|2]|05[6|8|9])+([0-9]{7})')
        isPhone = pattern_phone.search(url)
        if isPhone:
            check = True
            feature.append('phonenumberfeature')
        pattern_price = re.compile('(gia-)?([0-9])+(-ty|ty|tr|trieu|-tr|-trieu)')
        isPrice = pattern_price.search(url)
        if isPrice and (url.find('ban') or url.find('mua') or url.find('gia')):
            check = True
            feature.append('pricefeature')
        pattern_phongngu = re.compile('([1-9]-phong-ngu)|([1-9]phong)|([1-9]phong-ngu)|([1-9]phongngu)|([1-9]-phongngu)')
        isPhongNgu = pattern_phongngu.search(url)
        if isPhongNgu:
            check = True
            feature.append('phongngufeature')
        pattern_suface = re.compile('([0-9]*-m2)|([0-9]*m2)|([0-9]*mx[0-9]*m)|([0-9]*x[0-9]*m)|([1-9][0-9]*m)')
        isSuface = pattern_suface.search(url)
        if isSuface:
            check = True
            feature.append('dientichfeature')
        pattern_tang = re.compile('([1-9]*-tang)|([1-9]*tang)|([1-9][1-9]*t-)|([1-9]*lau)|([1-9]*-lau)|([1-9][1-9]*l-)')
        isTang = pattern_tang.search(url)
        if isTang:
            check = True
            feature.append('tangfeature')
        # loai bo list domain
        urlSplit = url.split("/")
        tmpURL = urlSplit[len(urlSplit) - 1]
        size = len(tmpURL.split('-'))
        if (size < 10) and (not check):
            feature.append('maybeList')

        # loại bỏ các đặc tính xấu hay bị lấn
        isTinDang = url.find('tin-dang')
        isMuaBan = url.find('mua-ban')
        if (isTinDang and isMuaBan) or (isMuaBan and (isPrice != None or isPhongNgu != None or isSuface != None or isPhone != None)):
            url.replace('mua-ban', 'ban')
        isCanHo = url.find('can-ho')
        if (isTinDang and isCanHo) or (isCanHo and (isPrice != None or isPhongNgu != None or isSuface != None or isPhone != None)):
            url.replace('can-ho', 'canho')

        return feature, check

    def extract_url(self, data):
        print('Parsing and cleaning URL ')
        #Parsing and cleaning URL
        self.features = []
        feature_text = data

        for t in feature_text:
            if type(t) != str:
                t = t.decode("UTF-8").encode('ascii', 'ignore')
            origin = t
            t = re.sub(r'[^a-zA-Z]', r' ', t)
            del_words = ['www', 'http', 'com', 'co', 'uk', 'org',
                         'https', 'html', 'ca', 'ee', 'htm',
                         'net', 'edu', 'index', 'asp', 'au', 'nz',
                         'txt', 'php', 'de', 'cgi', 'jp', 'hub',
                         'us', 'fr', 'webs', 'vn']
            stop_words = set([])
            stop_words.update(del_words)
            '''strip the words. and remove the stopwords in our URI strings'''
            text = (i.strip() for i in t.split())
            text = [t for t in text if t not in stop_words]
            text += self.extract_feature_url(origin);
            ''''join the words together'''
            text = " ".join(text)
            '''Append the result to the empty feature list. This would serve as our feauture
            for training and testing'''
            self.features.append(text)
        print('Done')
        print(20*'*')
        print('============== 100% COMPLETE ============')

        return self.features


    def extract_properties(self):
        print('Parsing and cleaning URL ')
        # Parsing and cleaning get features
        self.features = []
        for index, t in self.data.iterrows():
           text_feature = ""
           # xu ly url
           url_decode = self.convertURL(t['url'])
           title = t['title']
           description = t['description']
           if type(t['url']) != str:
               url_decode = t['url'].decode("UTF-8").encode('ascii', 'ignore')
           if type(t['title']) != str:
               title = t['title'].decode("UTF-8").encode('ascii', 'ignore')
           if type(t['description']) != str:
               description = t['description'].decode("UTF-8").encode('ascii', 'ignore')
           content = title + ' ' + description
           content = self.standraString(content)
           origin_url = url_decode
           url_decode = re.sub(r'[^a-zA-Z]', r' ', url_decode)
           origin_content = content
           origin_content = re.sub(r'[0-9]', r' ', origin_content)
           origin_content = re.sub(r'\s+', r' ', origin_content)

           del_words = ['www', 'http', 'com', 'co', 'uk', 'org',
                        'https', 'html', 'ca', 'ee', 'htm',
                        'net', 'edu', 'index', 'asp', 'au', 'nz',
                        'txt', 'php', 'de', 'cgi', 'jp', 'hub',
                        'us', 'fr', 'webs', 'vn']
           stop_words = set([])
           stop_words.update(del_words)
           '''strip the words. and remove the stopwords in our URI strings'''
           text = (i.strip() for i in url_decode.split())
           text = [t for t in text if t not in stop_words]
           properties_url, checkURL = self.extract_feature_url(origin_url)
           text += properties_url
           ''''join the words together'''
           text = " ".join(text)
           '''Join content'''
           text = text + ' ' + origin_content
           properties_content, checkContent = self.extract_feature_content(content)
           text = text + ' ' + " ".join(properties_content)
           if not checkContent and not checkURL:
               text = text + ' maybeList'
           '''Append the result to the empty feature list. This would serve as our feauture
            for training and testing'''
           self.features.append(text)

        print('Done')
        print(20*'*')
        print('============== 100% COMPLETE ============')
        return self.features

    def extract(self):
        print('Parsing and cleaning URL ')
        #Parsing and cleaning URL
        self.features = []
        feature_text = list(self.data['url'].values)
        for t in feature_text:
            if type(t) != str:
                t = t.decode("UTF-8").encode('ascii', 'ignore')
            origin = t
            t = re.sub(r'[^a-zA-Z]', r' ', t)

            '''you may want to include as many suffix as possible.
            This would have a positive effect on our prediction accuracy'''
            '''Another thing to do is to scrap all url suffix from Wiki'''

            del_words = ['www', 'http', 'com', 'co', 'uk', 'org',
                         'https', 'html', 'ca', 'ee', 'htm',
                         'net', 'edu', 'index', 'asp', 'au', 'nz',
                         'txt', 'php', 'de', 'cgi', 'jp', 'hub',
                         'us', 'fr', 'webs', 'vn']

            stop_words = set([])
            stop_words.update(del_words)

            '''strip the words. and remove the stopwords in our URI strings'''
            text = (i.strip() for i in t.split())
            text = [t for t in text if t not in stop_words]
            property_url, check = self.extract_feature_url(origin)
            text += property_url
            ''''join the words together'''
            text = " ".join(text)
            '''Append the result to the empty feature list. This would serve as our feauture
            for training and testing'''
            self.features.append(text)

        print('Done')
        print(20*'*')
        print('============== 100% COMPLETE ============')

        return self.features

#%% CONVERT CATEGORICAL DATA TO NUMERICS

class PREPROCESS(object):

    def __init__(self, X=None, Y=None):
        '''Arguments:
            X: feacture vector
                Y: scalar to be predicted
                    '''
        self.X = X
        self.Y = Y


    def loadTFIDFDictionary(self, pathVector='model/vectorizer-svm.sav', pathSelector='model/selector-svm.sav'):
        tfidf = pickle.load(open(pathVector, 'rb'))
        selector = pickle.load(open(pathSelector, 'rb'))
        return tfidf, selector;

    def processWithTFIDF(self, tfidf, selector):
        features_transformed = tfidf.transform(self.X)
        features_ = selector.transform(features_transformed).toarray()
        return features_


    def createTFIDFDictionary(self, save=False):
        vectorizer = TfidfVectorizer()
        features_train_transformed = vectorizer.fit_transform(self.X)
        selector = SelectPercentile(f_classif, percentile=25)
        selector.fit(features_train_transformed, self.Y)
        if save:
            pickle.dump(vectorizer, open('model/vectorizer-svm.sav', 'wb'))
            pickle.dump(selector, open('model/selector-svm.sav', 'wb'))
        return vectorizer, selector


    def process(self):
        '''Initialize TfidfVectorizer:
            TfidfVectorizer: Converst our categorical feature into numerical vectors
            '''
        vectorizer = TfidfVectorizer()
        features_train_transformed = vectorizer.fit_transform(self.X)
        print('Vectorizing completes....')
        print('Performing SelectPercentile completes....')
        '''SelectPercentile provides an automatic procedure for 
        keeping only a certain percentage of the best, associated features.
        f_classif: Used only for categorical targets and based on the 
        Analysis of Variance (ANOVA) statistical test.
        '''
        selector = SelectPercentile(f_classif, percentile=25)
        selector.fit(features_train_transformed, self.Y)
        features_train_ = selector.transform(features_train_transformed).toarray()

        print('SelectPercentile completes....')
        return features_train_, vectorizer, selector