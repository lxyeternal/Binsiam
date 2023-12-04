# -*- coding: UTF-8 -*-
"""
@Project ：Binsimi 
@File    ：Construct_Graph.py
@Author  ：honywen
@Date    ：2023/5/20 20:38 
@Software: PyCharm
"""

import os
import dgl
import pydot
import json

import torch
import torch as th
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from dgl import save_graphs, load_graphs
from codes.GenGraph.attribute import extract_attributes


class Dgl_Graph:
    def __init__(self):
        self.config_path = "config.json"
        with open(self.config_path, "r") as fr:
            self.config = json.load(fr)
        self.word2vec_model = self.config["word2vec_model_path"]
        self.operatordict = dict()
        self.node_list = list()
        self.bin_file_path = self.config["bindata_path"]
        self.embedding_size = self.config["embedding_size"]
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')


    @staticmethod  # 画图展示生成图的结构
    def ShowGraph(graph, node_dict, nodeLabel, EdgeLabel):
        plt.figure(figsize=(10, 10))
        G = graph.to_networkx(node_attrs=nodeLabel.split(), edge_attrs=EdgeLabel.split())  # 转换 dgl graph to networks
        pos = nx.spring_layout(G, k=5 * 1 / np.sqrt(len(G.nodes())), iterations=40)
        nx.draw(G, pos, edge_color="grey", node_size=500, with_labels=True)  # 画图，设置节点大小
        node_data = nx.get_node_attributes(G, nodeLabel)  # 获取节点的desc属性
        node_labels = {index: "" + node_dict[str(data)] for index, data in
                       enumerate(node_data)}  # 重新组合数据， 节点标签是dict, {nodeid:value,nodeid2,value2} 这样的形式
        pos_higher = {}

        for k, v in pos.items():  # 调整下顶点属性显示的位置，不要跟顶点的序号重复了
            if (v[1] > 0):
                pos_higher[k] = (v[0] - 0.04, v[1] + 0.04)
            else:
                pos_higher[k] = (v[0] - 0.04, v[1] - 0.04)
        nx.draw_networkx_labels(G, pos_higher, labels=node_labels, font_color="brown", font_size=12)  # 将desc属性，显示在节点上
        edge_labels = nx.get_edge_attributes(G, EdgeLabel)  # 获取边的weights属性，

        edge_labels = {(key[0], key[1]): "" + str(edge_labels[key].item()) for key in
                       edge_labels}  # 重新组合数据， 边的标签是dict, {(nodeid1,nodeid2):value,...} 这样的形式
        nx.draw_networkx_edges(G, pos, alpha=0.5)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12)  # 将Weights属性，显示在边上
        #     plt.savefig('test.jpg')
        plt.show()

    def transferedgetype(self, edge_type: str) -> int:
        vec_edge_type = 0
        if edge_type == '"CFG"':
            vec_edge_type = 1
        elif edge_type == '"DFG"':
            vec_edge_type = 2
        return vec_edge_type


    def code2vec(self, node_tokens):
        # load word2vec vectors
        word_vector = {}
        with open(self.word2vec_model, "r", encoding="utf8") as fr:
            for line in fr.readlines():
                line_list = line.strip().split(" ")
                word_vector[line_list[0]] = line_list[1:]
        node_attribute = []
        for node_token in node_tokens:
            if node_token in word_vector:
                node_attribute.append(word_vector[node_token])
            else:
                node_attribute.append(['0']*self.embedding_size)
        # Convert list of strings to numpy array of floats
        float_array = np.array(node_attribute, dtype=float)
        # Compute the mean of each column
        mean_columns = np.mean(float_array, axis=0)
        return mean_columns

    def gengraph(self, filename):
        self.node_list.clear()
        graph = pydot.graph_from_dot_file(filename)
        dot_graph = graph[0]
        graph_nodes = dot_graph.get_nodes()
        for node in graph_nodes:
            self.node_list.append(node.get_name())
        src_nodes = list()
        des_nodes = list()
        #  获取边的类型
        edge_types_list = list()
        #  生成图中的边及特征
        graph_edges = dot_graph.get_edges()
        for edge in graph_edges:
            src = edge.get_source()
            dst = edge.get_destination()
            src_index = self.node_list.index(src)
            dst_index = self.node_list.index(dst)
            src_nodes.append(src_index)
            des_nodes.append(dst_index)
            edge_attribute = edge.get_attributes()['label']
            vec_edge_type = self.transferedgetype(edge_attribute)
            edge_types_list.append(vec_edge_type)
        vec_edge_types = th.tensor(edge_types_list)
        u, v = th.tensor(src_nodes), th.tensor(des_nodes)
        # u, v = th.tensor(src_nodes).to(self.device), th.tensor(des_nodes).to(self.device)
        G = dgl.graph((u, v))
        G.edata['w'] = vec_edge_types
        # G.edata['w'] = vec_edge_types.to(self.device)
        #  生成图中的节点以及特征
        for node in graph_nodes:
            node_index = self.node_list.index(node.get_name())
            node_str_attributes = node.get_attributes()['label'][1:-1]
            print(node_str_attributes)
            node_str_attributes_split = node_str_attributes.split(' ')
            node_attributes_vec = self.code2vec(node_str_attributes_split)
            vec_attributes = th.tensor([node_attributes_vec], dtype=th.float32)
            #  add Statistical Features
            # statistical_feature = extract_attributes(node_str_attributes)
            # statistical_feature_tensor = th.tensor(statistical_feature).unsqueeze(0)
            # vec_attributes_concat = th.cat([vec_attributes, statistical_feature_tensor], dim=1)
            G.nodes[node_index].data['x'] = vec_attributes
        G = dgl.add_self_loop(G)
        G = G
        # G = G.to(device)
        return G


if __name__ == '__main__':
    dglgraph = Dgl_Graph()
    dglgraph.gengraph("/Users/blue/Desktop/findutils-4.9.0_clang-9.0_x86_64_O3_locate-re_search_internal.dot")
