# -*- coding: UTF-8 -*-
"""
@Project ：Binsimi 
@File    ：test.py
@Author  ：honywen
@Date    ：2023/8/1 22:58 
@Software: PyCharm
"""





def extract_attributes(assembly_code):
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



import networkx as nx

src_nodes = (1,2,2,1,4,4)
des_nodes = (2,5,3,4,3,5)

G_betweeness = nx.DiGraph(zip(src_nodes, des_nodes))
betweenness = nx.betweenness_centrality(G_betweeness)
print(betweenness)
# 计算offspring
offspring_dict = {n: G_betweeness.out_degree(n) for n in G_betweeness.nodes()}
print(offspring_dict)

import networkx as nx

src_nodes = (1, 2, 2, 1, 4, 4, 3)
des_nodes = (2, 5, 3, 4, 3, 5, 5)

G = nx.DiGraph()
G.add_edges_from(zip(src_nodes, des_nodes))

R = G.reverse()

offspring = dict()
for node in R.nodes():
    offspring[node] = len(nx.ancestors(R, node))

print(offspring)