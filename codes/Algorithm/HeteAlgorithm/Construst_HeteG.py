# -*- coding: UTF-8 -*-
"""
@Project ：Binsimi 
@File    ：Construst_Hete.py
@Author  ：honywen
@Date    ：2023/6/19 01:12 
@Software: PyCharm
"""


import os
import dgl
import pydot
import json
import torch as th
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from codes.GenGraph.attribute import extract_attributes


class HeteGraph:
    def __init__(self):
        self.data_transfer = {'mov', 'movapd', 'movaps', 'movd', 'movdqa', 'movq', 'movsd', 'movss', 'movsx', 'movzx', 'lea',
                         'push', 'pop', 'pushfd', 'popfd', 'vmovd', 'vmovdqa', 'vmovaps', 'vmovups', 'vmovq',
                         'vmovddup', 'vmovdqu', 'vpextrw', 'vpinsrd', 'vextracti128'}

        self.arithmetic_operations = {'add', 'sub', 'mul', 'div', 'inc', 'dec', 'neg', 'idiv', 'imul', 'addsd', 'subsd',
                                 'mulsd', 'divsd', 'addss', 'subss', 'mulss', 'divss', 'fadd', 'fsub', 'fmul', 'fdiv',
                                 'faddp', 'fsubp', 'fmulp', 'fdivp', 'fsubrp', 'fdivrp', 'fiadd', 'fist', 'fistp'}

        self.logical_operations = {'and', 'or', 'xor', 'not', 'andpd', 'orpd', 'xorpd', 'andps', 'orps', 'xorps', 'andnps',
                              'andnpd', 'por', 'vpxor', 'vpshufb', 'vpsubb', 'vpsadbw', 'vpslldq', 'pcmpeqd',
                              'vpcmpeqb'}

        self.comparison_and_jump = {'cmp', 'test', 'je', 'jne', 'jg', 'jl', 'jge', 'jle', 'ja', 'jb', 'jae', 'jbe', 'jns',
                               'js', 'jno', 'jpo', 'jpe', 'jmp', 'call', 'retn', 'cmpsd', 'cmpss', 'ucomiss', 'ucomisd',
                               'fcomi', 'fcomip', 'fucomi', 'fucomip', 'sete', 'setne', 'setg', 'setl', 'setge',
                               'setle', 'seta', 'setae', 'setb', 'setbe', 'seto', 'setpe', 'setpo', 'sets', 'setns',
                               'cmove', 'cmovne', 'cmovg', 'cmovl', 'cmovge', 'cmovle', 'cmova', 'cmovae', 'cmovb',
                               'cmovbe', 'cmovo', 'cmovno', 'cmovs', 'cmovns'}

        self.binary_operations = {'shl', 'shr', 'sar', 'ror', 'rol', 'shld', 'shrd', 'bswap', 'bts', 'bt', 'bsr', 'cwde',
                             'cdq'}

        self.floating_point_operations = {'fld', 'fld1', 'fldz', 'fldcw', 'fst', 'fstp', 'fnstcw', 'fnstsw', 'fchs', 'fxch',
                                     'fxam', 'fcmove', 'fcmovne', 'fcmovb', 'fcmovbe', 'fcmovnbe', 'fcmove', 'fcmovbe',
                                     'fcmovb', 'fcmovnbe', 'fcmovne', 'cvtsi2sd', 'cvtss2sd', 'cvtsd2ss', 'cvttss2si',
                                     'cvttsd2si', 'vzeroupper', 'rep', 'repne'}

        self.other = {'nop', 'cpuid', 'leave', 'entry_point', 'vpclmulqdq'}
        self.node_type_list = ["other", "data_transfer", "arithmetic_operations", "logical_operations", "comparison_and_jump", "binary_operations", "floating_point_operations"]
        self.base_dotfile_path = "../../../dataset/Dotfile/"
        self.all_node_instructions = list()
        self.config_path = "../../GenGraph/config.json"
        with open(self.config_path, "r") as fr:
            self.config = json.load(fr)
        self.word2vec_model = self.config["word2vec_model_path"]
        self.embedding_size = self.config["embedding_size"]
        self.operatordict = dict()
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')


    def parse_node_instructions(self) -> None:
        dotfiles = os.listdir(self.base_dotfile_path)
        for dotfile in dotfiles:
            graph = pydot.graph_from_dot_file(os.path.join(self.base_dotfile_path, dotfile))
            dot_graph = graph[0]
            graph_nodes = dot_graph.get_nodes()
            for node in graph_nodes:
                node_str_attributes = node.get_attributes()['label'][1:-1]
                node_str_attributes_split = node_str_attributes.split(' ')
                if len(node_str_attributes_split) != 0:
                    self.all_node_instructions.append(node_str_attributes_split[0])


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
                node_attribute.append(['0'] * self.embedding_size)
        # Convert list of strings to numpy array of floats
        float_array = np.array(node_attribute, dtype=float)
        # Compute the mean of each column
        mean_columns = np.mean(float_array, axis=0)
        return mean_columns


    def transfer_edgetype(self, edge_type: str) -> int:
        vec_edge_type = 0
        if edge_type == '"CFG"':
            vec_edge_type = 1
        elif edge_type == '"DFG"':
            vec_edge_type = 2
        return vec_edge_type


    def transfer_nodetype(self, node_instruction: str) -> int:
        instruction_type_list = [self.other, self.data_transfer, self.arithmetic_operations, self.logical_operations, self.comparison_and_jump,
                                 self.binary_operations, self.floating_point_operations]
        for instruction_type_index in range(len(instruction_type_list)):
            if node_instruction in instruction_type_list[instruction_type_index]:
                node_label = self.node_type_list[instruction_type_index]
                break
            else:
                node_label = self.node_type_list[0]
        return node_label


    def gengraph(self, filename):
        graph = pydot.graph_from_dot_file(filename)
        dot_graph = graph[0]
        graph_nodes = dot_graph.get_nodes()
        #  extract all node in graph and represent as the instruction type
        node_attrtype_dict = dict()
        node_type_dict = dict()
        node_type_list = []
        for node in graph_nodes:
            node_index = node.get_name()
            node_str_attributes = node.get_attributes()['label'][1:-1]
            node_str_attributes_split = node_str_attributes.split(' ')
            node_type = self.transfer_nodetype(node_str_attributes_split[0])
            node_vec_attributes = self.code2vec(node_str_attributes_split)
            node_vec_attributes = th.tensor([node_vec_attributes], dtype=th.float32)
            #  add Statistical Features
            statistical_feature = extract_attributes(node_str_attributes)
            statistical_feature_tensor = th.tensor(statistical_feature).unsqueeze(0)
            vec_attributes_concat = th.cat([node_vec_attributes, statistical_feature_tensor], dim=1)
            node_attrtype_dict[node_index] = (node_type, vec_attributes_concat)
            node_type_dict[node_index] = node_type
        #   class the node
        for o in self.node_type_list:
            keys = [k for k, v in node_type_dict.items() if v == o]
            if keys:
                node_type_list.append(keys)
            else:
                node_type_list.append([])
        #   attr
        node_attr_list = list()
        for node_type in node_type_list:
            tmp_list = list()
            for node_name in node_type:
                node_type_attr = node_attrtype_dict[node_name][1]
                tmp_list.append(node_type_attr)
            node_attr_list.append(tmp_list)
        #  extract all edge and node, classify the different type edge and insert the tuple
        edge_type_list = list()
        edge_attr_list = list()
        graph_edges = dot_graph.get_edges()
        for edge in graph_edges:
            src = edge.get_source()
            dst = edge.get_destination()
            src_node_type = node_attrtype_dict[src][0]
            dst_node_type = node_attrtype_dict[dst][0]
            edge_attribute = edge.get_attributes()['label']
            vec_edge_type = self.transfer_edgetype(edge_attribute)
            edge_attr_list.append((src, vec_edge_type, dst))
            edge_type_list.append((src_node_type, vec_edge_type, dst_node_type))
        #  find all same type tuple
        noduplicates_edge_type_list = list(set(edge_type_list))
        edge_srcdst_node = list()
        for i in range(len(noduplicates_edge_type_list)):
            edge_srcdst_node.append(([], []))
        for edge_attr_index in range(len(edge_attr_list)):
            src_node_type = edge_type_list[edge_attr_index][0]
            dst_node_type = edge_type_list[edge_attr_index][2]
            #  shenzhibuqingle //
            src_node = edge_attr_list[edge_attr_index][0]
            dst_node = edge_attr_list[edge_attr_index][2]
            src_node_index = node_type_list[self.node_type_list.index(src_node_type)].index(src_node)
            dst_node_index = node_type_list[self.node_type_list.index(dst_node_type)].index(dst_node)
            #  insert
            edge_srcdst_node[noduplicates_edge_type_list.index(edge_type_list[edge_attr_index])][0].append(src_node_index)
            edge_srcdst_node[noduplicates_edge_type_list.index(edge_type_list[edge_attr_index])][1].append(dst_node_index)
        #  gen hete graph
        hete_graph_data = dict()
        for edge_srcdst_index in range(len(edge_srcdst_node)):
            edge_type = noduplicates_edge_type_list[edge_srcdst_index]
            hete_graph_data[edge_type] = (th.tensor(edge_srcdst_node[edge_srcdst_index][0]), th.tensor(edge_srcdst_node[edge_srcdst_index][1]))
        G = dgl.heterograph(hete_graph_data).to(self.device)
        #  add the attr of node
        for node_type_index in range(len(node_attr_list)):
            if node_attr_list[node_type_index] == []:
                continue
            node_type = self.node_type_list[node_type_index]
            G.nodes[node_type].data['hy'] = th.stack(node_attr_list[node_type_index])
        return G


    def start(self):
        dotfiles = os.listdir(self.base_dotfile_path)[:100]
        for dotfile in dotfiles:
            dotfile_fullpath = os.path.join(self.base_dotfile_path, dotfile)
            G = self.gengraph(dotfile_fullpath)
            print(G)



if __name__ == '__main__':
    hetegraph = HeteGraph()
    hetegraph.start()










