# -*- coding: UTF-8 -*-
"""
@Project ：few_shot_learning 
@File    ：fasttext_model.py
@Author  ：honywen
@Date    ：2022/12/29 15:41 
@Software: PyCharm
"""

import json
from gensim.models import FastText
from preprocess_data import PreprocessData


class FasttextModel:
    def __init__(self):
        self.config_path = "config.json"
        with open(self.config_path, "r") as fr:
            self.config = json.load(fr)
        self.traindata = list()


    def load_data(self):
        preprecessdata = PreprocessData()
        preprecessdata.read_data()
        self.traindata = preprecessdata.raw_data


    def trainvec(self):
        self.load_data()
        model = FastText(self.traindata,
                         vector_size = self.config['embedding_size'],
                         window = self.config['window'],
                         min_count = self.config['min_count'],
                         min_n = self.config['min_n'],
                         max_n = self.config['max_n'],
                         word_ngrams = self.config['word_ngrams'],
                         alpha = self.config['alpha'])
        # model.save(self.config['word2vec_model_path'])
        model.wv.save_word2vec_format(self.config['fasttext_model_path'], binary=False)


if __name__ == '__main__':
    fasttextmodel = FasttextModel()
    fasttextmodel.trainvec()
