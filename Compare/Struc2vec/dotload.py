# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# @File     : dotload.py
# @Project  : Binsiam
# Time      : 2023/12/31 01:31
# Author    : honywen
# version   : python 3.8
# Description：
"""

import torch
import pydot
from torch_geometric.data import Data


def gengraph(filename):
    node_list = []
    graph = pydot.graph_from_dot_file(filename)
    dot_graph = graph[0]
    graph_nodes = dot_graph.get_nodes()
    for node in graph_nodes:
        node_list.append(node.get_name())
    src_nodes = list()
    des_nodes = list()
    #  获取边的类型
    #  生成图中的边及特征
    graph_edges = dot_graph.get_edges()
    for edge in graph_edges:
        src = edge.get_source()
        dst = edge.get_destination()
        src_index = node_list.index(src)
        dst_index = node_list.index(dst)
        src_nodes.append(src_index)
        des_nodes.append(dst_index)

    #  生成图中的节点以及特征
    nodes_features = []
    for node in graph_nodes:
        node_str_attributes = node.get_attributes()['label'][1:-1]
        node_attributes_vec = eval(node_str_attributes)
        nodes_features.append(node_attributes_vec)
    data = Data(x=torch.tensor(nodes_features), edge_index=torch.tensor([src_nodes, des_nodes]))
    return data