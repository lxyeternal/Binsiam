# -*- coding: UTF-8 -*-
"""
@Project ：Binsimi 
@File    ：parsedot.py
@Author  ：honywen
@Date    ：2023/5/20 20:16 
@Software: PyCharm
"""

import pydot


graph = pydot.graph_from_dot_file("../../dataset/Dotfile/output.dot")
dot_graph = graph[0]

graph_nodes = dot_graph.get_nodes()
for node in graph_nodes:
    print(node.get_attributes()['label'][1:-1])


graph_edges = dot_graph.get_edges()
for edge in graph_edges:
    src = edge.get_source()
    dst = edge.get_destination()
    print(src)
    print(dst)
    print(edge.get_attributes()['label'])
