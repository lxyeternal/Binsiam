# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# @File     : filter_files.py
# @Project  : Binsimi
# Time      : 2023/10/23 22:58
# Author    : honywen
# version   : python 3.8
# Description：
"""


import os
import shutil
from codes.Preprocess.makepair import parse_samplename


def fileter_main():
    source_dir = "/Users/blue/Documents/Binsimi/dataset/Rawdata/BinKit_normal"
    target_dir = "/Users/blue/Documents/Binsimi/dataset/Rawdata"
    project_options = ['binutils', 'coreutils', 'findutils', 'recutils']
    compiler_options = ['gcc-6.5.0', 'clang-9.0']
    architecture_options = ['x86_64', 'arm_64']
    optimization_options = ['O0', 'O1', 'O2', 'O3']
    for project in project_options:
        target_project_dir = os.path.join(target_dir, project)
        if not os.path.exists(target_project_dir):
            os.makedirs(target_project_dir)
        project_binfiles = os.listdir(os.path.join(source_dir, project))
        flag = 0
        for project_binfile in project_binfiles:
            binfile_parse = parse_samplename(project_binfile)
            if (binfile_parse["compiler"] in compiler_options) \
                and (binfile_parse["artitecture"] in architecture_options) \
                and (binfile_parse["optimizer"] in optimization_options):
                if (flag == 0) and (project == "coreutils"):
                    flag = 1
                else:
                    shutil.copyfile(os.path.join(source_dir, project, project_binfile), os.path.join(target_project_dir, project_binfile))
                    flag = 0

fileter_main()