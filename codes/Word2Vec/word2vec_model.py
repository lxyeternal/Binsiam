# -*- coding: UTF-8 -*-
"""
@Project ：few_shot_learning 
@File    ：word2vec_model.py
@Author  ：honywen
@Date    ：2022/12/29 15:40 
@Software: PyCharm
"""


import json
from gensim.models import word2vec
from preprocess_data import PreprocessData


class Word2vecModel:
    def __init__(self):
        self.config_path = "config.json"
        with open(self.config_path, "r") as fr:
            self.config = json.load(fr)
        self.traindata = list()


    def load_data(self):
        preprecessdata = PreprocessData()
        preprecessdata.read_data()
        self.traindata = preprecessdata.raw_data
        #  该函数的目的是把训练数据处理成以下形式
        # [['int', 'a', '=', '1'], ['void', 'static', 'func', '(', ')']]


    def trainvec(self):
        self.load_data()
        model = word2vec.Word2Vec(self.traindata,
                                  workers = self.config['workers'],
                                  vector_size = self.config['embedding_size'],
                                  min_count = self.config['min_count'],
                                  window = self.config['window'],
                                  sample = self.config['sample'])
        # model.save("word2vec.model")
        model.wv.save_word2vec_format(self.config['word2vec_model_path'], binary=False)


if __name__ == '__main__':
    word2vecmodel = Word2vecModel()
    word2vecmodel.trainvec()


