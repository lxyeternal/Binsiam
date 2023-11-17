# -*- coding: UTF-8 -*-
"""
@Project ：few_shot_learning 
@File    ：glove_model.py
@Author  ：honywen
@Date    ：2022/12/29 15:41 
@Software: PyCharm
"""


import json
from glove import Glove
from glove import Corpus
from preprocess_data import PreprocessData


class GloveModel:
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
        corpus_model = Corpus()
        corpus_model.fit(self.traindata, window=self.config['window'])
        print('Dict size: %s' % len(corpus_model.dictionary))
        print('Collocations: %s' % corpus_model.matrix.nnz)
        glove = Glove(no_components = self.config['embedding_size'],
                      learning_rate = self.config['learning_rate'])  # no_components 维度，可以与word2vec一起使用。
        glove.fit(corpus_model.matrix,
                  epochs = self.config['epochs'],
                  no_threads = self.config['no_threads'],
                  verbose = True)
        glove.add_dictionary(corpus_model.dictionary)

        # 3.glove模型保存与加载
        corpus_model.save(self.config['glove_model_path'])
        # corpus_model = Corpus.load('corpus.model')
        # 指定词条词向量
        print(glove.word_vectors[glove.dictionary['int']])


if __name__ == '__main__':
    glovemodel = GloveModel()
    glovemodel.trainvec()