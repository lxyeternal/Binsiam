# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# @File     : json2dot.py
# @Project  : Binsimi
# Time      : 2023/10/14 21:10
# Author    : honywen
# version   : python 3.8
# Description：
"""

import os
import json
from codes.Preprocess.makepair import parse_samplename


class JsonToDot:
    def __init__(self, json_filepath, normalize_task, represent_type, binfilename, dot_file_path):
        self.jsondata = None
        self.function_name = ""
        self.dot_file_path = dot_file_path
        self.binfilename = binfilename.split("/")[-1]
        self.normalize_task = normalize_task
        self.json_filepath = json_filepath
        self.represent_type = represent_type
        self.name_parse_result = parse_samplename(self.binfilename)
        self.output_file = ""

    def json_to_dot(self):
        dir_path = os.path.join(self.dot_file_path, self.name_parse_result["Project"], self.represent_type+"-"+self.normalize_task)
        self.output_file = dir_path + '/' + self.binfilename + "-" + self.function_name + '.dot'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        if self.jsondata is None:
            print("No data to write. Please load data first.")
            return
        with open(self.output_file, 'w') as f:
            # Write the header
            f.write('digraph "func" {\n')
            # Write all nodes
            for node in self.jsondata['nodes']:
                f.write('"{}" [label = "{}" ]\n'.format(node['id'], node['label']))
            # Write all edges
            for i, adj_list in enumerate(self.jsondata['links']):
                f.write('"{}" -> "{}" [ label = "{}" ]\n'.format(adj_list['source'], adj_list['target'], adj_list['label']))
            # Write the footer
            f.write('}\n')

    def json2dot_main(self, function_name):
        self.function_name = function_name
        with open(os.path.join(self.json_filepath), 'r') as f:
            self.jsondata = json.load(f)
        self.json_to_dot()
