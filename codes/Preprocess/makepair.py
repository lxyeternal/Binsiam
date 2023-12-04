# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# @File     : makepair.py
# @Project  : Binsimi
# Time      : 2023/10/14 16:05
# Author    : honywen
# version   : python 3.8
# Description：
"""

import os
import json
import torch
from tqdm import tqdm, trange
from collections import Counter
from dgl import save_graphs, load_graphs
from multiprocessing import Pool, cpu_count
from codes.GenGraph.Construct_Graph import Dgl_Graph


def parse_samplename(samplename: str):
    # Extract main parts
    parts = samplename.split('_')
    project = parts[0]
    compiler = parts[1]
    architecture = parts[2] + "_" + parts[3]
    optimizer = parts[4]
    file_name = parts[5][:-1]
    try:
        # Determine file_name and function_name
        split_by_function = samplename.split("-")
        function_name = split_by_function[-1].replace(".dot", "")
    except:
        function_name = None

    # Return results
    return {
        "project": project,
        "compiler": compiler,
        "artitecture": architecture,
        "optimizer": optimizer,
        "filename": file_name,
        "function_name": function_name
    }


class MakePairs:
    def __init__(self, project_options, cross_project, present_type, cross_compiler, cross_optimizer,
                 cross_architecture, normalize_task):
        self.present_type = present_type
        self.cross_project = cross_project
        self.cross_compiler = cross_compiler
        self.cross_optimizer = cross_optimizer
        self.cross_architecture = cross_architecture
        self.normalize_task = normalize_task
        self.project_options = project_options
        self.config_path = "./config.json"
        with open(self.config_path, "r") as fr:
            self.config = json.load(fr)
        self.bindata_path = self.config["bindata_path"]
        self.dotfile_path = self.config["dot_file_path"]
        self.pairdata_path = self.config["pairdata_path"]
        self.dgldata_path = self.config["dgldata_path"]
        self.dglgraph = Dgl_Graph()
        self.task_path = os.path.join('_'.join(self.project_options), str(self.normalize_task) + "-" + self.present_type)
        self.dataset_filename = '_'.join(self.project_options) + "-" + str(self.normalize_task) + "-" + self.present_type + "-" + str(self.cross_project) + "-" + str(self.cross_compiler) + "-" + str(self.cross_optimizer) + "-" + str(self.cross_architecture)
        self.save_graphs_filename = self.dataset_filename + '.bin'
        self.pos_threshold = 40000
        self.neg_threshold = 40000
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def write_pairs(self, paired_dotfiles, txtfile):
        with open(txtfile, 'w', encoding='utf-8') as file:
            for item in paired_dotfiles:
                file.write(f"{item}\n")


    def read_pairs(self, txtfile):
        with open(txtfile, 'r', encoding='utf-8') as file:
            file_content = file.readlines()
        unique_lines = list(set(line.strip() for line in file_content))
        return unique_lines


    def list_dotfiles(self):
        alldots = dict()
        for project in self.project_options:
            project_dir = os.path.join(self.dotfile_path, project, str(normalize_task) + "-" + present_type)
            project_dotfiles = os.listdir(project_dir)
            for project_dotfile in project_dotfiles:
                dotfile_fullpath = os.path.join(project_dir, project_dotfile)
                dotfile_info = parse_samplename(project_dotfile)
                dot_type = ''
                if not cross_project:
                    dot_type += dotfile_info["project"] + "-"
                if not cross_compiler:
                    dot_type += dotfile_info["compiler"] + "-"
                if not cross_architecture:
                    dot_type += dotfile_info["artitecture"] + "-"
                if not cross_optimizer:
                    dot_type += dotfile_info["optimizer"] + "-"
                if dot_type not in alldots:
                    alldots[dot_type] = []
                alldots[dot_type].append(dotfile_fullpath)
        return alldots

    def single_process_graph_vec(self, paired_dotfiles_subset):
        graph_vecs = {}
        for paired_dotfile in tqdm(paired_dotfiles_subset):
            try:
                dgl_vec = self.dglgraph.gengraph(paired_dotfile)
                graph_vecs[paired_dotfile] = dgl_vec
            except Exception as e:
                # Optionally print the error for logging purposes
                print(f"Error processing {paired_dotfile}: {str(e)}")
                # Continue processing the next file in the subset
                continue
        return graph_vecs


    def graph_vec(self, paired_graphs):
        num_processes = 50  # 获取CPU核心数作为进程数，或者你可以自行设置
        with Pool(num_processes) as pool:
            # 将paired_graphs分割为num_processes个子列表
            chunk_size = len(paired_graphs) // num_processes
            results = pool.map(self.single_process_graph_vec,
                               [paired_graphs[i:i + chunk_size] for i in range(0, len(paired_graphs), chunk_size)])
            # 合并所有子进程返回的结果字典
        graph_vecs = {}
        for res in results:
            graph_vecs.update(res)
        return graph_vecs


    def make_pairs(self, data_dict):
        paired_dotfiles = []
        labels = []
        pos_count = 0
        neg_count = 0
        for key, files in data_dict.items():
            # Iterate over each combination of dot files
            for i in tqdm(range(len(files))):
                for j in range(i + 1, len(files)):
                    details1 = parse_samplename(files[i])
                    details2 = parse_samplename(files[j])
                    # Check the criteria for pairing
                    same_project = details1["project"] == details2["project"]
                    same_compiler = details1["compiler"] == details2["compiler"]
                    same_optimizer = details1["optimizer"] == details2["optimizer"]
                    same_architecture = details1["artitecture"] == details2["artitecture"]
                    same_file = details1["filename"] == details2["filename"]
                    same_function = details1["function_name"] == details2["function_name"]

                    # Pairing logic based on conditions
                    if (self.cross_project ^ same_project) and \
                            (self.cross_compiler ^ same_compiler) and \
                            (self.cross_optimizer ^ same_optimizer) and \
                            (self.cross_architecture ^ same_architecture):

                        if same_file and same_function:
                            if pos_count < self.pos_threshold:
                                paired_dotfiles.extend([files[i], files[j]])
                                labels.append(1)
                                pos_count += 1
                        # Pairing dissimilar graphs
                        elif not same_file or not same_function:
                            if neg_count < self.neg_threshold:
                                paired_dotfiles.extend([files[i], files[j]])
                                labels.append(0)
                                neg_count += 1
        print(f"Total number of pairs: {len(paired_dotfiles)}")
        deduped_paired_dotfiles = list(set(paired_dotfiles))
        print(f"Total number of pairs: {len(deduped_paired_dotfiles)}")
        dot_vecs_dict = self.graph_vec(deduped_paired_dotfiles)
        new_graphs = [dot_vecs_dict[key] for key in paired_dotfiles]
        count = Counter(labels)
        print(count, len(labels))
        new_labels = torch.tensor(labels)
        save_graphs(os.path.join(self.bindata_path, self.save_graphs_filename), new_graphs, {'labels': new_labels})

    def makepairmain(self, project_dots_dict):
        self.make_pairs(project_dots_dict)
        # unique_lines = self.read_pairs(os.path.join(self.pairdata_path, self.dataset_filename+".txt"))
        # self.graph_vec(unique_lines)


if __name__ == '__main__':
    cross_project = False
    cross_compiler = False
    cross_optimizer = True
    cross_architecture = False
    normalize_task = True
    present_type = "CSG"
    project_options = ['findutils-4.9.0']
    if len(project_options) > 2:
        cross_project = True
    makepairs = MakePairs(project_options, cross_project, present_type, cross_compiler, cross_optimizer,
                          cross_architecture, normalize_task)
    project_dots_dict = makepairs.list_dotfiles()
    makepairs.makepairmain(project_dots_dict)
