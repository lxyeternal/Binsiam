# -*- coding: UTF-8 -*-
"""
@Project ：Binsimi 
@File    ：bulidacfg.py
@Author  ：honywen
@Date    ：2023/7/17 17:10 
@Software: PyCharm
"""



import os
import dgl
import pydot
import json
import torch as th
import networkx as nx
from dgl import save_graphs, load_graphs


class ACFG_Graph:
    def __init__(self):
        self.config_path = "config.json"
        with open(self.config_path, "r") as fr:
            self.config = json.load(fr)
        self.operatordict = dict()
        self.node_list = list()
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')


    def brandes_betweenness(self, graph):
        # 初始化节点的betweenness字典
        betweenness_dict = {node: 0.0 for node in graph.nodes()}

        for node in graph.nodes():
            # 以当前节点为起点计算所有节点的最短路径
            shortest_paths = nx.single_source_shortest_path(graph, node)

            # 初始化中间值
            num_sp = {node: 0 for node in graph.nodes()}
            dependency = {node: 0.0 for node in graph.nodes()}
            stack = []

            # 计算所有节点的依赖值
            while stack:
                parent, child = stack[-1]
                if child not in shortest_paths[parent]:
                    stack.pop()
                    if child in graph:
                        for ancestor in stack:
                            dependency[ancestor] += (1 + dependency[child]) / num_sp[child] * num_sp[ancestor]
                else:
                    child = stack.pop()
                    for parent in stack:
                        num_sp[parent] += num_sp[child]
                        dependency[parent] += (1 + dependency[child]) / num_sp[child] * num_sp[parent]

            # 累加每个节点的betweenness值
            for node in graph.nodes():
                if node != node:
                    betweenness_dict[node] += dependency[node]

        return betweenness_dict


    def offspring_count(self, src_nodes, des_nodes):
        G = nx.DiGraph()
        G.add_edges_from(zip(src_nodes, des_nodes))
        R = G.reverse()
        offspring = dict()
        for node in R.nodes():
            offspring[node] = len(nx.ancestors(R, node))
        return offspring


    def extract_attributes(self, assembly_code):
        # 初始化属性计数器
        string_constants = 0
        numeric_constants = 0
        transfer_instructions = 0
        calls = 0
        instructions = 0
        arithmetic_instructions = 0

        # 将汇编代码按行拆分成指令列表
        instructions_list = assembly_code.strip().split("',")

        # 遍历每条指令
        for instruction in instructions_list:
            instruction = instruction.strip()[1:]
            # 假设指令以空格分隔操作码和操作数
            # 先使用空格分隔
            result = instruction.split()
            # 再使用逗号分隔
            opcodes = [elem.strip(',') for elem in result]
            for opcode in opcodes:
                opcode = opcode.strip()
                # 统计字符串常量和数值常量
                if '0x' in opcode:
                    string_constants += 1
                elif opcode.isnumeric():
                    numeric_constants += 1
                # 统计跳转指令
                if opcode in ('jmp', 'je', 'jne', 'jg', 'jl', 'jge', 'jle', 'ja', 'jb', 'jae', 'jbe'):
                    transfer_instructions += 1
                # 统计调用指令
                if opcode == 'call':
                    calls += 1
                # 统计总指令数
                instructions += 1
                # 统计算术指令
                if opcode in ('add', 'sub', 'mul', 'div', 'inc', 'dec', 'neg', 'idiv', 'imul'):
                    arithmetic_instructions += 1
        return [string_constants, numeric_constants, transfer_instructions, calls, instructions, arithmetic_instructions]

    def gengraph(self, filename):
        self.node_list.clear()
        graph = pydot.graph_from_dot_file(filename)
        dot_graph = graph[0]
        graph_nodes = dot_graph.get_nodes()
        for node in graph_nodes:
            self.node_list.append(node.get_name())
        graph_edges = dot_graph.get_edges()
        src_nodes = list()
        des_nodes = list()
        for edge in graph_edges:
            src = edge.get_source()
            dst = edge.get_destination()
            src_nodes.append(src)
            des_nodes.append(dst)
        #  calculate the graph betweenness
        G_betweeness = nx.DiGraph(zip(src_nodes, des_nodes))
        betweenness = nx.betweenness_centrality(G_betweeness)
        # 计算offspring
        offspring_dict = self.offspring_count(src_nodes, des_nodes)
        #  生成图中的边及特征
        src_nodes_index = list()
        des_nodes_index = list()
        for edge in graph_edges:
            src = edge.get_source()
            dst = edge.get_destination()
            src_index = self.node_list.index(src)
            dst_index = self.node_list.index(dst)
            src_nodes_index.append(src_index)
            des_nodes_index.append(dst_index)
        u, v = th.tensor(src_nodes_index), th.tensor(des_nodes_index)
        G = dgl.graph((u, v))
        #  生成图中的节点以及特征
        for node in graph_nodes:
            node_name = node.get_name()
            node_index = self.node_list.index(node_name)
            node_str_attributes = node.get_attributes()['label'][1:-1]
            node_offspring = offspring_dict[node_name]
            node_betweenness = betweenness[node_name]
            #  add Statistical Features
            statistical_feature = self.extract_attributes(node_str_attributes)
            statistical_feature = statistical_feature + [node_offspring, node_betweenness]
            vec_attributes_concat = th.tensor([statistical_feature])
            G.nodes[node_index].data['x'] = vec_attributes_concat
        G = dgl.add_self_loop(G)
        G = G
        return G


    def start(self):
        graph_data = []
        for file in os.listdir(self.config["dot_file_path"])[:10]:
            dotfile = os.path.join(self.config["dot_file_path"], file)
            G = self.gengraph(dotfile)
            # graph_data.append(G)
        # return graph_data


    def save(self, graph_data, graph_labels):
        # 保存图和标签
        graph_path = os.path.join("saved path")
        save_graphs(graph_path, graph_data, {'labels': graph_labels})



if __name__ == '__main__':
    dglgraph = ACFG_Graph()
    dglgraph.start()