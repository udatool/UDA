import re
import matplotlib.pyplot as plt
import base64
import binascii
import json
from tqdm import tqdm


arm_jmp_inst = ['tbz', 'cbnz', 'b.eq', 'b.ne', 'tbnz', 'cbz', 'b.vs', 'b.le', 'b', 'b.vc', 'b.mi', 'b.gt', 'b.hs', 'b.lo', 'b.ge', 'bl', 'b.pl', 'b.hi', 'b.ls', 'b.lt', 'bpl', 'bmi', 'blt', 'bne', 'bhs', 'bvc', 'bvs', 'blhs', 'blne', 'beq', 'ble', 'blo', 'bhi', 'bls', 'bleq', 'bgt', 'bge']
mips_jmp_inst = ['bgez', 'beq', 'bc3tl', 'bc1f', 'b', 'bc1t', 'blez', 'bne', 'beqz', 'bnez', 'bc3t', 'bgtz', 'bc3fl', 'bltz', 'bal', 'bc3f', 'j', 'jal']


def trans_opt_to_arch_id(opt):
    if 'x86' in opt:
        return 0
    if 'arm' in opt:
        return 1
    if 'mips' in opt:
        return 2
    print(f'[!]unknown arch:{opt}')
    return -1


def trans_opt_to_bits_id(opt):
    if '32' in opt:
        return 0
    if '64' in opt:
        return 1
    print(f'[!]unknown arch:{opt}')
    return -1


def calculate_base_address(pc_address):
    # 将PC寄存器地址的后三位置0
    base_address = pc_address & ~0xFFF  # 将最后12位清零
    return base_address


def hex_subtraction(hex_str1, hex_str2):
    # 将十六进制字符串转换为整数
    int_value1 = int(hex_str1, 16) if type(hex_str1) != int else hex_str1
    int_value2 = int(hex_str2, 16) if type(hex_str2) != int else hex_str2

    # 执行减法操作
    result = int_value1 - int_value2

    # 将结果转换回十六进制字符串
    result_hex = hex(result)

    return result_hex


def is_float(num_str):
    flag = False
    try:
        reg = re.compile(r'^[0-9]+\.[0-9]+$')
        res = reg.match(str(num_str))
        if res:
            flag = True
    except Exception as ex:
        print("is_float() - error: " + str(ex))
    return flag


def is_value(value_str):
    flag = False
    try:
        reg = re.compile(r'^v[0-9]+\..*$')
        res = reg.match(str(value_str))
        if res:
            flag = True
    except Exception as ex:
        print("is_float() - error: " + str(ex))
    return flag


def tokenize_instruction(ins):
    """
    Tokenize the instruction in input.

    Args
        ins: a string representin an assemly instruction

    Return
        list: a list of tokens.
    """
    ins = ins.replace(',', ' , ')
    ins = ins.replace('[', ' [ ')
    ins = ins.replace(']', ' ] ')
    ins = ins.replace(':', ' : ')
    ins = ins.replace('*', ' * ')
    ins = ins.replace('(', ' ( ')
    ins = ins.replace(')', ' ) ')
    ins = ins.replace('{', ' { ')
    ins = ins.replace('}', ' } ')
    ins = ins.replace('#', '')
    ins = ins.replace('$', '')
    ins = ins.replace('!', ' ! ')
    ins = re.sub(r'-(0[xX][0-9a-fA-F]+)', r'- \1', ins)
    ins = re.sub(r'-([0-9a-zA-Z]+)', r'- \1', ins)
    return ins.split()


def norm_inst(instruction_dict, hex_list, arch=None, split_hex=True, prefix=True, jmp=True):
    code_list = list()
    inst_length_list = list()
    for bb_index, bb_addr in enumerate(instruction_dict):
        bb_hex = hex_list[bb_index]
        bb_list = instruction_dict[bb_addr]
        for inst_index, inst in enumerate(bb_list):
            inst_list = list()
            tokens = tokenize_instruction(inst)
            for token_index, token in enumerate(tokens):
                token = token.lower()
                if '0x' in token:
                    if split_hex:
                        if arch == 'x86':
                            if jmp:
                                if tokens[0][0] == 'j' or tokens[0] == 'call':
                                    inst_hex = bb_hex[inst_index]

                                    inst_hex_list = inst_hex.split()
                                    if inst_hex_list[0] == '0f':
                                        inst_hex_list = inst_hex_list[2:]
                                    else:
                                        inst_hex_list = inst_hex_list[1:]
                                    inst_hex_str = ' '.join(inst_hex_list)
                                    inst_hex_str = add_hex_prefix(inst_hex_str, arch, prefix)
                                    inst_list.extend(inst_hex_str.split())
                                else:
                                    inst_list.extend(hex_num_split(token, arch, prefix))
                            elif tokens[0][0] == 'j' or tokens[0] == 'call':
                                inst_list.append('hexvar')
                            else:
                                inst_list.extend(hex_num_split(token, arch, prefix))
                        elif arch == 'arm':
                            if jmp:
                                curr_addr = int(bb_addr) + inst_index * 4
                                if tokens[0] == 'adrp':
                                    base_addr = calculate_base_address(curr_addr)
                                    dis_hex = hex_subtraction(token, base_addr)
                                    inst_list.extend(hex_num_split(dis_hex, arch, prefix))
                                elif tokens[0] == 'adr':
                                    dis_hex = hex_subtraction(token, curr_addr)
                                    inst_list.extend(hex_num_split(dis_hex, arch, prefix))
                                elif tokens[0] in arm_jmp_inst and len(token) > 4:
                                    token_integer = int(token, 16)
                                    dis = token_integer - int(curr_addr)
                                    if dis < 0:
                                        dis = (1 << 32) + dis
                                        dis_hex = format(dis, '08x')
                                    else:
                                        dis_hex = hex(dis)
                                    inst_list.extend(hex_num_split(dis_hex, arch, prefix))
                                else:
                                    inst_list.extend(hex_num_split(token, arch, prefix))
                            else:
                                inst_list.extend(hex_num_split(token, arch, prefix))
                        elif arch == 'mips':
                            if jmp:
                                # if (inst_index == len(bb_list) - 2 or tokens[0][0] == 'j' or tokens[0][0] == 'b') and len(token) > 6:
                                if tokens[0] in mips_jmp_inst:
                                    inst_hex = bb_hex[inst_index]
                                    if len(tokens) == 2:
                                        inst_hex_list = inst_hex.split()[:3]
                                    elif len(tokens) > 2:
                                        inst_hex_list = inst_hex.split()[:2]
                                    else:
                                        inst_hex_list = []
                                        print(tokens)
                                    inst_hex_str = ' '.join(inst_hex_list)
                                    inst_hex_str = add_hex_prefix(inst_hex_str, arch, prefix)
                                    inst_list.extend(inst_hex_str.split())
                                elif len(token) > 6:
                                    print(tokens)
                                else:
                                    inst_list.extend(hex_num_split(token, arch, prefix))
                            else:
                                inst_list.extend(hex_num_split(token, arch, prefix))
                    else:
                        inst_list.append('hexvar')
                elif token.isdigit():
                    if split_hex:
                        inst_list.extend(hex_num_split(token, arch, prefix))
                    else:
                        inst_list.append('num')
                elif is_float(token):
                    inst_list.append('float')
                elif is_value(token):
                    inst_list.append('value')
                else:
                    inst_list.append(token)
            inst_str = ' '.join(inst_list)
            if prefix:
                inst_str = add_prefix(inst_str, arch)
            code_list.append(inst_str)
            inst_length_list.append(len(inst_str.split()))
    return code_list, inst_length_list


def hex_num_split(hex_num, arch, prefix=True):
    if hex_num.startswith('0x'):
        hex_num = hex_num[2:]
    elif hex_num.startswith('-0x'):
        hex_num = hex_num[3:]
    if len(hex_num) % 2 != 0:
        hex_num = '0' + hex_num
    hex_num_str = ' '.join([hex_num[i:i + 2] for i in range(0, len(hex_num), 2)])
    hex_num_str = add_hex_prefix(hex_num_str, arch, prefix)
    hex_num_list = hex_num_str.split()
    hex_num_list.reverse()
    return hex_num_list


def norm_hex(hex_list, arch, prefix=True):
    norm_hex_list = list()
    inst_length_list = list()
    for hex_bb in hex_list:
        for hex_inst in hex_bb:
            hex_inst = add_hex_prefix(hex_inst, arch, prefix)
            norm_hex_list.append(hex_inst)
            inst_length_list.append(len(hex_inst.split()))
    return norm_hex_list, inst_length_list


def add_prefix(sent, arch):
    prefix = arch[0] + '_'
    words = [prefix + word for word in sent.split()]
    for i in range(len(words)):
        if words[i].startswith(arch[0] + '_' + arch[0] + '_'):
            words[i] = words[i][0] + words[i][2:]
        if words[i].startswith(arch[0] + '_' + arch[0] + arch[0] + '_'):
            words[i] = words[i][2:]
    sent = " ".join(words)
    return sent


def add_hex_prefix(sent, arch=None, pre=True):
    if pre:
        prefix = arch[0] + arch[0] + '_'
    else:
        prefix = '#'
    words = [prefix + word for word in sent.split()]
    sent = " ".join(words)
    return sent


def generate_position_encodings(ins_len_lists, max_length=512):
    """
    Generate instruction position encoding and operator position encoding based on the length of each instruction.

    :param ins_len_lists: List containing the length of each instruction.
    :return: Tuple of (instruction_position_encoding, operator_position_encoding)
    """
    all_ins_pos = []
    all_op_pos = []

    # Iterate over each instruction and its length
    for ins_len_str in ins_len_lists:
        ins_pos = []
        op_pos = []
        ins_len_list = list(map(int, ins_len_str.split()))
        for ins_index, ins_length in enumerate(ins_len_list):
            for op_index in range(int(ins_length)):
                ins_pos.append(ins_index)
                op_pos.append(op_index)
                if len(ins_pos) == max_length:  # Check if the length exceeds 512
                    break  # Exit the inner loop
            if len(ins_pos) == max_length:  # Check again as we're now outside the inner loop
                break  # Exit the outer loop
        # ins_pos.extend([0] * (max_length - len(ins_pos)))
        # op_pos.extend([0] * (max_length - len(op_pos)))
        all_ins_pos.append(ins_pos)
        all_op_pos.append(op_pos)
    return all_ins_pos, all_op_pos
