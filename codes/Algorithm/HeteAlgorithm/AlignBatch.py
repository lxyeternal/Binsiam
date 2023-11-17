# -*- coding: UTF-8 -*-
"""
@Project ：Binsimi 
@File    ：consistbatch.py
@Author  ：honywen
@Date    ：2023/6/19 04:55 
@Software: PyCharm
"""

import dgl
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#  对每一个batch保持边缘类型保持一致
def alignbatch(graph_batch):
    all_edge_types = list()
    align_batch_graph = list()
    for one_graph in graph_batch:
        all_edge_types = all_edge_types + one_graph.canonical_etypes
    all_edge_types_set = list(set(all_edge_types))
    for graph in graph_batch:
        cpudevice = torch.device('cpu')
        graph = graph.to(cpudevice)
        new_graph_data = dict()
        one_graph_edge_types = graph.canonical_etypes
        for edge_type in all_edge_types_set:
            if edge_type not in one_graph_edge_types:
                #  将该边缘类型添加到图中
                new_graph_data[edge_type] = ([], [])
        for etype in one_graph_edge_types:
            edge_ids = torch.tensor([i for i in range(graph.num_edges(etype))])
            edge_nodes = graph.find_edges(edge_ids, etype)
            new_graph_data[etype] = edge_nodes
        new_graph = dgl.heterograph(new_graph_data)
        #  将节点特征赋值重新赋值给新图
        #  节点类型
        ntypes = graph.ntypes
        print(graph.etypes)
        new_graph.edges['1'].data['ac'] = torch.randn(new_graph.num_edges('1'), 1)
        new_graph.edges['2'].data['ac'] = torch.randn(new_graph.num_edges('2'), 1)
        for ntype in ntypes:
            new_graph.nodes[ntype].data['hy'] = graph.nodes[ntype].data['hy']
        align_batch_graph.append(new_graph.to(device))
    return align_batch_graph
