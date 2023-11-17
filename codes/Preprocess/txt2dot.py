# -*- coding: UTF-8 -*-
"""
@Project ：Binsimi 
@File    ：txt2dot.py
@Author  ：honywen
@Date    ：2023/5/20 20:32 
@Software: PyCharm
"""

import json
import os


class DotGenerator:
    def __init__(self):
        self.data = None
        self.config_path = "config.json"
        with open(self.config_path, "r") as fr:
            self.config = json.load(fr)

    def load_from_txt(self, file_path):
        with open(os.path.join(self.config["cfg_file_path"], file_path), 'r') as f:
            data_str = f.read()
            self.data = json.loads(data_str)

    def write_to_dot(self, output_path):
        if self.data is None:
            print("No data to write. Please load data first.")
            return

        with open(os.path.join(self.config["dot_file_path"], output_path), 'w') as f:
            # Write the header
            f.write('digraph "func" {\n')

            # Write all nodes
            for node in self.data['nodes']:
                f.write('"{}" [label = "{}" ]\n'.format(node['id'], node['text']))

            # Write all edges
            for i, adj_list in enumerate(self.data['adjacency']):
                for adj in adj_list:
                    f.write('"{}" -> "{}" [ label = "CFG" ]\n'.format(self.data['nodes'][i]['id'], adj['id']))

            # Write the footer
            f.write('}\n')


if __name__ == '__main__':
    dot_generator = DotGenerator()
    dot_generator.load_from_txt('cfg.txt')
    dot_generator.write_to_dot('output.dot')