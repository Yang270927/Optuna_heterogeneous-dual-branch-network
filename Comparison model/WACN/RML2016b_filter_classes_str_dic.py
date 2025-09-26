import numpy as np
import torch
import torch.nn.functional as F
# # 检查可用的 GPU 数量
# num_gpus = torch.cuda.device_count()
# print("Number of available GPUs: ", num_gpus)
# # 使用所有可用的 GPU
# devices = [torch.device(f'cuda:{i}') for i in range(num_gpus)]
# print("Using devices:", devices)
label_mapping = {'16QAM': 0, '64QAM': 1, '8PSK': 2, 'AM-DSB-SC': 3, 'AM-SSB-SC': 4, 'BPSK': 5, 'GMSK': 6,'QPSK':7}
labels = {
    0 : ("16QAM",0),
    1: ("64QAM", 1),
    2: ("8PSK", 2),
    3: ("AM-DSB-SC",3),
    4: ("AM-SSB-SC",4),
    5: ("8PSK", 5),
    6: ("GMSK", 6),
    7: ("QPSK",7),
}

def get_modulation_types(keys):
    return [labels[key][0] for key in keys]

def get_positions(keys):
    return [labels[key][1] for key in keys]

def convert_labels_to_dict(label_array, label_dict=labels):
    dict_labels = {i: label_dict[label][0] for i, label in enumerate(label_array)}
    return dict_labels


def tensor_to_labels(tensor, label_dict=labels):
    tensor = tensor.flatten().long().tolist()  # 将输入Tensor转换为list
    labels = [label_dict[i][1] for i in tensor]  # 获取对应的位置值
    return torch.tensor(labels)  # 将位置值转换为Tensor并返回
