import numpy as np
from elftools.elf.elffile import ELFFile
import argparse
import os
from tqdm import tqdm
import json
from model_utils import UDA, BertCustomModel
from transformers import AutoTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader
from data_utils import trans_opt_to_arch_id, trans_opt_to_bits_id


def add_prefix(sent):
    prefix = '#'
    words = [prefix + word.lower() for word in sent.split()]
    sent = " ".join(words)
    return sent


def split_sentence(sentence, max_length):
    sentences = [' '.join(sentence[i:i + max_length]) for i in range(0, len(sentence), max_length)]
    return sentences


def split_addr(sentence, max_length):
    sentences = [sentence[i:i + max_length] for i in range(0, len(sentence), max_length)]
    return sentences


def do_label(model, dataloader):
    model.eval()
    model.cuda()

    pred_label_list = []
    for hex_seq, hex_mask, arch_ids, bits_ids in dataloader:
        with torch.no_grad():
            outputs = model(input_ids=hex_seq, attention_mask=hex_mask, arch_ids=arch_ids, bits_ids=bits_ids)
            logits = outputs.logits
            preds = torch.argmax(logits, -1).cpu()
            pred_label_list.extend(preds[:, 1:-1])
    return pred_label_list


def addr_map(pred_label_list, addr_list, addr_mapping):
    preds_list = []
    assert len(pred_label_list) == len(addr_list)
    for index in range(len(pred_label_list)):
        if len(pred_label_list[index]) != len(addr_list[index]):
            preds_batch = pred_label_list[index][:len(addr_list[index])]
        else:
            preds_batch = pred_label_list[index]
        preds_list.append(preds_batch)
    preds_all_list = torch.cat(preds_list, dim=0).tolist()
    start_addr_list = []
    end_addr_list = []
    for i, (label, addr) in enumerate(zip(preds_all_list, addr_mapping)):
        if label == 0:
            start_addr_list.append(int(addr))
        elif label == 1:
            end_addr_list.append(int(addr))

    return start_addr_list, end_addr_list


def hex_map(pred_label_list, hex_list):
    preds_list = torch.cat(pred_label_list, dim=0).tolist()
    if len(preds_list) != len(hex_list):
        preds_list = preds_list[:len(hex_list)]
    # Resulting list to store the groups
    result = []
    current_group = []
    # Iterate through both lists
    for i, (label, hex_value) in enumerate(zip(preds_list, hex_list)):
        if label == 0:
            if current_group:
                result.append(current_group)  # Save the previous group if it exists
            current_group = [hex_value]  # Start a new group
            # Check if the next label is also 0, meaning a group of length 1
            if i + 1 < len(preds_list) and preds_list[i + 1] == 0:
                result.append(current_group)  # Append group immediately
                current_group = []  # Reset group
        elif label == 1:
            current_group.append(hex_value)
            result.append(current_group)
            current_group = []  # Reset group after finishing
        else:
            current_group.append(hex_value)
    return result


def _get_elf_code(path):
    with open(path, 'rb') as f:
        elf = ELFFile(f)
        bits = str(elf.elfclass)
        machine = elf['e_machine']
        arch = None
        if machine == 'EM_386' or machine == 'EM_X86_64':
            arch = 'x86'
        elif machine == 'EM_ARM' or machine == 'EM_AARCH64':
            arch = 'arm'
        elif machine == 'EM_MIPS':
            arch = 'mips'

        code_data = []
        code_addr = []
        code_section = elf.get_section_by_name('.text')
        offset = code_section['sh_addr']
        code_data.append(code_section.data())
        code_addr.append((offset, len(code_data[-1])))
        return b''.join(code_data), code_addr, arch, bits


def main():
    parser = argparse.ArgumentParser(description='Example')
    parser.add_argument('--bin_path', default="./bin_example/a2ps-4.14_clang-4.0_arm_32_O0_a2ps.elf",
                        help='Path to the binary to disassemble')
    parser.add_argument('--output_dir', help='Path to save', default="./output/")
    parser.add_argument('--func_model_path', default="./model_saved/uda_function",
                        help='Function boundary model path.')
    parser.add_argument('--inst_model_path', default="./model_saved/uda_instruction",
                        help='Instruction boundary model path.')
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument("--max_length", default=512, type=int, help="Max length")
    args = parser.parse_args()
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.func_model_path, use_fast=False,
                                              do_lower_case=False, do_basic_tokenize=False)
    model_inst = UDA.from_pretrained(args.inst_model_path)
    model_func = UDA.from_pretrained(args.func_model_path)

    code, code_addr, arch, bits = _get_elf_code(args.bin_path)

    arch_id = trans_opt_to_arch_id(arch)
    bits_id = trans_opt_to_bits_id(bits)

    hex_str = code.hex()
    formatted_hex_str = ' '.join(hex_str[i:i + 2] for i in range(0, len(hex_str), 2))
    formatted_hex_str = add_prefix(formatted_hex_str)
    hex_list = formatted_hex_str.split()

    addr_mapping = np.empty(len(code), dtype=np.uint64)
    cur_idx = 0
    for addr, length in code_addr:
        addr_mapping[cur_idx:cur_idx + length] = np.arange(start=addr, stop=addr + length, step=1, dtype=np.int64)
        cur_idx += length

    hex_sents_list = split_sentence(hex_list, args.max_length - 2)
    addr_list = split_addr(addr_mapping, args.max_length - 2)

    hex_token_list = []
    hex_mask_list = []
    arch_ids_list = []
    bits_ids_list = []
    for hex_data in hex_sents_list:
        hex_token = tokenizer(hex_data, add_special_tokens=True, max_length=args.max_length,
                              padding='max_length', truncation=True, return_tensors='pt')
        hex_token_list.append(hex_token['input_ids'])
        hex_mask_list.append(hex_token['attention_mask'])
        arch_ids_list.append(torch.tensor([arch_id] * args.max_length).unsqueeze(0))
        bits_ids_list.append(torch.tensor([bits_id] * args.max_length).unsqueeze(0))
    hex_seq_tensor = torch.cat(hex_token_list, dim=0).cuda()
    hex_mask_tensor = torch.cat(hex_mask_list, dim=0).cuda()
    arch_ids_tensor = torch.cat(arch_ids_list, dim=0).cuda()
    bits_ids_tensor = torch.cat(bits_ids_list, dim=0).cuda()

    dataset_hex = TensorDataset(hex_seq_tensor, hex_mask_tensor, arch_ids_tensor, bits_ids_tensor)
    dataloader_hex = DataLoader(dataset_hex, batch_size=args.batch_size)

    inst_label_list = do_label(model_inst, dataloader_hex)
    func_label_list = do_label(model_func, dataloader_hex)

    inst_start_addr_list, inst_end_addr_list = addr_map(inst_label_list, addr_list, addr_mapping)
    func_start_addr_list, func_end_addr_list = addr_map(func_label_list, addr_list, addr_mapping)

    inst_hex = hex_map(inst_label_list, hex_list)
    func_hex = hex_map(func_label_list, hex_list)

    bin_name = os.path.basename(args.bin_path)
    save_name = bin_name + '.json'
    save_path = os.path.join(args.output_dir, save_name)

    data = {'func_start_addr': func_start_addr_list, 'func_end_addr': func_end_addr_list}

    with open(save_path, 'w') as file:
        json.dump(data, file)


if __name__ == '__main__':
    main()
