# -*- coding: UTF-8 -*-
"""
@Project ：Binsimi
@File    ：parsebin.py
@Author  ：honywen
@Date    ：2023/5/8 01:32
@Software: PyCharm
"""

import angr
import networkx as nx
import matplotlib.pyplot as plt
from graphviz import Source
import pyvex
from networkx.drawing.nx_pydot import write_dot


class BinaryAnalysis:
    def __init__(self, binary_path):
        self.proj = angr.Project(binary_path, auto_load_libs=False)
        self.cfg = self.proj.analyses.CFGFast()
        # self.cfg = self.proj.analyses.CFGAccurate(function_names=function_names)


    def list_functions(self):
        functions = []
        for func in self.cfg.functions.values():
            functions.append(func.name)
        return functions

    # 绘制并保存所有函数的AST、CFG和DFG
    def visualize_all_functions(self):
        complete_ast = nx.DiGraph()
        complete_cfg = nx.DiGraph()
        complete_dfg = nx.DiGraph()

        for function in self.cfg.functions.values():
            # AST
            ast_graph = function.transition_graph

            # Quote node names and attributes containing colons
            for node in ast_graph.nodes(data=True):
                node_name = str(node[0])
                if ':' in node_name:
                    ast_graph = nx.relabel_nodes(ast_graph, {node[0]: f'"{node_name}"'})

                for attr_key, attr_value in node[1].items():
                    if ':' in attr_key:
                        node[1][f'"{attr_key}"'] = node[1].pop(attr_key)

                    if ':' in str(attr_value):
                        node[1][attr_key] = f'"{attr_value}"'

            complete_ast = nx.compose(complete_ast, ast_graph)

            # CFG
            cfg_graph = function.graph
            complete_cfg = nx.compose(complete_cfg, cfg_graph)

            # DFG
            dfg = nx.DiGraph()
            for block in function.blocks:
                try:
                    irsb = block.vex
                except angr.errors.SimTranslationError:
                    print(f"Unable to translate bytecode at block {block.addr:x}")
                    continue

                for stmt_idx, stmt in enumerate(irsb.statements):
                    if not isinstance(stmt, pyvex.IRStmt.WrTmp):
                        continue
                    dst = stmt.tmp
                    srcs = [stmt.data]
                    dfg.add_node(dst)
                    for src in srcs:
                        if isinstance(src, pyvex.IRExpr.RdTmp):
                            dfg.add_edge(src.tmp, dst)

        # 保存所有函数的AST
        write_dot(complete_ast, "all_functions_ast.dot")
        src_ast = Source.from_file("all_functions_ast.dot")
        src_ast.view()

        # 保存所有函数的CFG
        write_dot(complete_cfg, "all_functions_cfg.dot")
        src_cfg = Source.from_file("all_functions_cfg.dot")
        src_cfg.view()

        # 保存所有函数的DFG
        nx.draw(complete_dfg, with_labels=True)
        plt.savefig("all_functions_dfg.png")
        plt.show()

    def visualize_ast(self, function_name):
        function = self.cfg.functions.get(function_name)
        if function is not None:
            graph = function.transition_graph

            # Check and modify node names and attributes that contain a colon
            for node in graph.nodes(data=True):
                node_name = str(node[0])
                # Check if the node name contains a colon
                if ':' in node_name:
                    # Replace the node name with the double-quoted version
                    graph = nx.relabel_nodes(graph, {node[0]: f'"{node_name}"'})

                # Check if any of the node attributes contain a colon
                for attr_key, attr_value in node[1].items():
                    if ':' in attr_key:
                        # Replace the attribute key with the double-quoted version
                        node[1][f'"{attr_key}"'] = node[1].pop(attr_key)

                    if ':' in str(attr_value):
                        # Replace the attribute value with the double-quoted version
                        node[1][attr_key] = f'"{attr_value}"'

            dot_file = f"{function_name}_ast.dot"
            write_dot(graph, dot_file)
            src = Source.from_file(dot_file)
            src.view()

    def visualize_cfg(self, function_name):
        function = self.cfg.functions.get(function_name)
        if function is None:
            print("Function not found.")
            return
        graph = function.graph
        pos = nx.spring_layout(graph)
        labels = {node: hex(node.addr) for node in graph.nodes}
        nx.draw_networkx_nodes(graph, pos)
        nx.draw_networkx_edges(graph, pos)
        nx.draw_networkx_labels(graph, pos, labels=labels)
        plt.axis('off')
        plt.show()

    def visualize_dfg(self, function_name):
        function = self.cfg.functions.get(function_name)
        if function is None:
            print("Function not found.")
            return
        dfg = nx.DiGraph()
        for block in function.blocks:
            irsb = block.vex
            for stmt_idx, stmt in enumerate(irsb.statements):
                if not isinstance(stmt, pyvex.IRStmt.WrTmp):
                    continue
                dst = stmt.tmp
                srcs = [stmt.data]
                dfg.add_node(dst)
                for src in srcs:
                    if isinstance(src, pyvex.IRExpr.RdTmp):
                        dfg.add_edge(src.tmp, dst)
        nx.draw(dfg, with_labels=True)
        plt.show()

if __name__ == '__main__':
    # 使用方法
    binary_path = '/Users/blue/Documents/Binsimi/data/gnu_debug_sizeopt/gnu_debug/a2ps/a2ps-4.14_clang-4.0_arm_32_Os_fixnt.elf'  # 替换为实际二进制文件路径
    analysis = BinaryAnalysis(binary_path)
    functions = analysis.list_functions()

    # 可以先绘制整个elf文件的AST、CFG、DFG
    analysis.visualize_all_functions()

    # for function_name in functions:
    #     print(f"Visualizing AST, CFG, and DFG for function: {function_name}")
    #     analysis.visualize_ast(function_name)
    #     analysis.visualize_cfg(function_name)
    #     analysis.visualize_dfg(function_name)


