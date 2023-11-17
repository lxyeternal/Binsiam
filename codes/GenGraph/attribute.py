# -*- coding: UTF-8 -*-
"""
@Project ：Binsimi 
@File    ：attribute.py
@Author  ：honywen
@Date    ：2023/7/12 13:55 
@Software: PyCharm
"""


def extract_attributes(asm_code):
    arithmetic_inst = ['add', 'sub', 'mul', 'div', 'inc', 'dec', 'neg', 'idiv', 'imul', 'addsd', 'subsd', 'mulsd',
                       'divsd', 'addss', 'subss', 'mulss', 'divss', 'fadd', 'fsub', 'fmul', 'fdiv', 'faddp',
                       'fsubp', 'fmulp', 'fdivp', 'fsubrp', 'fdivrp', 'fiadd', 'fist', 'fistp']
    logic_inst = ['and', 'or', 'xor', 'not', 'andpd', 'orpd', 'xorpd', 'andps', 'orps', 'xorps', 'andnps', 'andnpd',
                  'por', 'vpxor']
    shift_inst = ['shl', 'shr', 'sar', 'ror', 'rol', 'shld', 'shrd']
    data_inst = ['mov', 'movapd', 'movaps', 'movd', 'movdqa', 'movq', 'movsd', 'movss', 'movsx', 'movzx', 'lea',
                 'push', 'pop', 'pushfd', 'popfd', 'vmovd', 'vmovdqa', 'vmovaps', 'vmovups', 'vmovq', 'vmovddup',
                 'vmovdqu', 'vpextrw', 'vpinsrd', 'vextracti128']
    call_inst = ['call']
    jump_inst = ['je', 'jne', 'jg', 'jl', 'jge', 'jle', 'ja', 'jb', 'jae', 'jbe', 'jns', 'js', 'jno', 'jpo', 'jpe','jmp']

    arithmetic_count = 0
    logic_count = 0
    shift_count = 0
    data_count = 0
    call_count = 0
    jump_count = 0
    references_count = 0

    for code_token in asm_code.split(" "):
        inst = code_token.strip()
        if inst in arithmetic_inst:
            arithmetic_count += 1
        elif inst in logic_inst:
            logic_count += 1
        elif inst in shift_inst:
            shift_count += 1
        elif inst in data_inst:
            data_count += 1
        elif inst in call_inst:
            call_count += 1
        elif inst in jump_inst:
            jump_count += 1
        # 统计引用次数
        elif inst in ["[", "address"]:
            references_count += 1

    results = [arithmetic_count, logic_count, shift_count, data_count, call_count, jump_count, references_count]

    return results
