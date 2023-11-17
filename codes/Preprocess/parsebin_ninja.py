# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# @File     : parsebin_ninja.py
# @Project  : Binsimi
# Time      : 2023/10/14 17:25
# Author    : honywen
# version   : python 3.8
# Description：
"""


import networkx
import networkx as nx
from binaryninja import *
from json2dot import JsonToDot
from networkx.readwrite import json_graph


class ProcessBinaryFile:
    def __init__(self, project, binfilename, represent_type, normalize_task):
        self.tmp_jsonpath = "tmp.json"
        self.represent_type = represent_type
        self.normalize_task = normalize_task
        self.binfilename = binfilename
        self.tmp_jsonpath = "../../Jsondata/{}.json".format(self.binfilename.split("/")[-1])
        self.opcode_path = "../../Opcodes/{}-opcode.txt".format(self.binfilename.split("/")[-1])
        self.bv = BinaryViewType.get_view_of_file(self.binfilename)
        self.config_path = "config.json"
        with open(self.config_path, "r") as fr:
            self.config = json.load(fr)
        self.corpus_filename = project + '_' + represent_type + '_' + str(normalize_task) + '.txt'
        self.json2dot_inst = JsonToDot(self.tmp_jsonpath, self.represent_type, str(normalize_task), self.binfilename, self.config["dot_file_path"])
        self.corpus_filepath = os.path.join(self.config["corpus_path"], self.corpus_filename)

    @staticmethod
    def parse_instruction_fine(ins, symbol_map, string_map, opcode_path):
        # 对每条指令进行规范化
        ins = re.sub('\s+', ' ', ins, 1)
        ins = ins.replace(',', ' ,')
        parts = ins.split(' ')
        operand = []
        if len(parts) > 1:
            operand = parts[1:]
        for i in range(len(operand)):
            symbols = re.split('([0-9A-Za-z]+)', operand[i])
            for j in range(len(symbols)):
                if symbols[j][:2] == '0x' and len(symbols[j]) >= 6:
                    if int(symbols[j], 16) in symbol_map:
                        symbols[j] = "symbol"
                    elif int(symbols[j], 16) in string_map:
                        symbols[j] = "string"
                    else:
                        symbols[j] = "address"
                if symbols[j][:3] == '#0x' and len(symbols[j]) < 6 or symbols[j][:2] == '0x' and len(symbols[j]) < 6:
                    symbols[j] = "address"
                # symbols[j] = filter_reg(symbols[j])
            operand[i] = ' '.join(symbols)
            if i == len(operand) - 1:
                operand[i] = operand[i].rstrip()

        opcode = parts[0]
        with open(opcode_path, 'a') as f:
            f.write(opcode)
            f.write(' ')
        ins = ' '.join([opcode] + operand)
        ins = re.sub(r'\s+', ' ', ins)
        return ins

    def function_to_cfg(self, func):
        """
        Generate the CFG for a function
        :param func:
        :return:
        """
        G = nx.DiGraph()
        for block in func:
            print(block.start)
            curr = block.start
            predecessor = curr
            for inst in block:
                G.add_node(curr, label=self.bv.get_disassembly(curr))
                print(self.bv.get_disassembly(curr))
                # 判断当前节点是否是一个基本块的开始
                if curr != block.start:
                    G.add_edge(predecessor, curr, label='CFG')
                predecessor = curr
                curr += inst[1]

            for edge in block.outgoing_edges:
                G.add_edge(predecessor, edge.target.start, label='CFG')
        return G

    def function_to_dfg(self, func):
        """
        Generate the DFG for a function
        :param func:
        :return:
        """
        G = nx.DiGraph()
        # # 入口节点
        # G.add_node(-1, label='entry_point')
        # 对于该函数的每个基本块，遍历其mlil表示每个指令，mlil为binary ninja的中间表示
        # print("------------------00",func.name)
        for block in func.mlil:
            mlil_blocks = []
            blocks = []
            for ins in block:
                # 对于每个指令，将其地址添加到图中作为一个节点，并设置该节点的文本为该指令的汇编表示
                G.add_node(ins.address, label=self.bv.get_disassembly(ins.address))
                # depd存储边信息
                depd = []
                mlil_blocks.append(ins)
                blocks.append(self.bv.get_disassembly(ins.address))
                # 对于读变量，查看定义该变量的指令并将其作为父节点，以建立数据依赖边
                for var in ins.vars_read:
                    depd = [(func.mlil[i].address, ins.address)
                            # func.mlil.get_var_definitions(var)获取定义此变量的所有MLIL指令的索引列表
                            for i in func.mlil.get_var_definitions(var)
                            if func.mlil[i].address != ins.address]
                # 对于写变量，查看使用该变量的指令并将其作为该指令的子节点，以建立数据依赖边
                for var in ins.vars_written:
                    depd += [(ins.address, func.mlil[i].address)
                             # func.mlil.get_var_uses(var)获取使用此变量的所有MLIL指令的索引列表
                             for i in func.mlil.get_var_uses(var)
                             if func.mlil[i].address != ins.address]
                if depd:
                    G.add_edges_from(depd, label='DFG')
        return G


    def normalize_graph(self, graph_inst):
        symbol_map = {}
        string_map = {}
        for sym in self.bv.get_symbols():
            symbol_map[sym.address] = sym.full_name
        for string in self.bv.get_strings():
            string_map[string.start] = string.value
        isolated_nodes = list(nx.isolates(graph_inst))
        graph_inst.remove_nodes_from(isolated_nodes)
        # 规范化
        for n in graph_inst:
            if n != -1 and graph_inst.nodes[n]['label'] is not None:
                graph_inst.nodes[n]['label'] = ProcessBinaryFile.parse_instruction_fine(
                    graph_inst.nodes[n]['label'], symbol_map, string_map, self.opcode_path)
                with open(self.corpus_filepath, 'a') as f:
                    f.write(graph_inst.nodes[n]['label'])
                    f.write(' ')
        with open(self.corpus_filepath, 'a') as f:
            f.write('\n')
        return graph_inst


    def process_file(self):
        # 获取二进制文件中所有函数的名称
        for func in self.bv.functions:
            if self.represent_type == "CFG":
                graph_inst = self.function_to_cfg(func)
            elif self.represent_type == "DFG":
                graph_inst = self.function_to_dfg(func)
            else:   # CSG
                graph_inst = networkx.compose(self.function_to_cfg(func), self.function_to_dfg(func))
            if len(graph_inst.nodes) > 2:
                # normalize step
                if self.normalize_task:
                    graph_inst = self.normalize_graph(graph_inst)
                else:
                    for n in graph_inst:
                        if n != -1 and graph_inst.nodes[n]['label'] is not None:
                            with open(self.corpus_filepath, 'a') as f:
                                f.write(graph_inst.nodes[n]['label'])
                                f.write(' ')
                    with open(self.corpus_filepath, 'a') as f:
                        f.write('\n')
                    return graph_inst
                json_data = json_graph.node_link_data(graph_inst)
                with open(self.tmp_jsonpath, "w") as json_file:
                    json.dump(json_data, json_file)
            self.json2dot_inst.json2dot_main(func.name)


if __name__ == "__main__":
    source_dir = "../../dataset/Rawdata"
    project_options = ['coreutils', 'findutils', 'recutils', 'binutils']
    log_file_path = "processed_binfiles.txt"
    with open(log_file_path, 'a') as file:
        for project in project_options:
            binfiles = os.listdir(os.path.join(source_dir, project))
            for id, binfile in enumerate(binfiles):
                print("{}-----{}------{}".format(project, str(id), binfile))
                # try:
                processbinaryfile = ProcessBinaryFile(project, os.path.join(source_dir, project, binfile), "CSG", True)
                processbinaryfile.process_file()
                file.write(f"{project},{binfile},Success\n")
                # except:
                #     file.write(f"{project},{binfile},Error\n")
                #     pass