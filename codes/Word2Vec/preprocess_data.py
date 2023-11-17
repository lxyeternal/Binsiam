# -*- coding: UTF-8 -*-
"""
@Project ：few_shot_learning 
@File    ：preprocess_data.py
@Author  ：honywen
@Date    ：2022/12/29 15:41 
@Software: PyCharm
"""


class PreprocessData:
    def __init__(self):
        self.data_file = "../../dataset/Corpus/disassembly.txt"
        self.raw_data = list()


    def read_data(self):
        f = open(self.data_file, 'r')
        token_lines = f.readlines()
        for token_line in token_lines:
            token_line_split = token_line.strip().split(" ")
            self.raw_data.append(token_line_split)


# preprecessdata = PreprocessData()
# preprecessdata.read_data()
