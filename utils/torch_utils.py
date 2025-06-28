"""
Utility functions for torch.
"""

import torch
import torch.nn.functional as F
import numpy as np
import random
import math


def get_optimizer(name, parameters, lr, l2=0):
    if name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, weight_decay=l2)
    elif name == 'adam':
        return torch.optim.Adam(parameters, weight_decay=l2)
    elif name == 'adamax':
        return torch.optim.Adamax(parameters, weight_decay=l2)
    elif name == 'adadelta':
        return torch.optim.Adadelta(parameters, lr=lr, weight_decay=l2)
    else:
        raise Exception("Unsupported optimizer: {}".format(name))


def change_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def flatten_indices(seq_lens, width):
    flat = []
    for i, l in enumerate(seq_lens):
        for j in range(l):
            flat.append(i * width + j)
    return flat


def set_cuda(var, cuda):
    if cuda:
        return var.cuda()
    return var


def keep_partial_grad(grad, topk, finetune_epoch=-1, epoch=-1):
    """
    Keep only the topk rows of grads.
    """
    assert topk < grad.size(0)
    if 0 <= finetune_epoch <= epoch:
        return grad
    else:
        grad.data[topk:].zero_()
        return grad


def save(model, optimizer, opt, filename):
    params = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': opt
    }
    try:
        torch.save(params, filename)
    except BaseException:
        print("[ Warning: model saving failed. ]")


def load(model, optimizer, filename):
    try:
        dump = torch.load(filename)
    except BaseException:
        print("[ Fail: model loading failed. ]")
    if model is not None:
        model.load_state_dict(dump['model'])
    if optimizer is not None:
        optimizer.load_state_dict(dump['optimizer'])
    opt = dump['config']
    return model, optimizer, opt


def load_config(filename):
    try:
        dump = torch.load(filename, map_location='cpu')
    except BaseException:
        print("[ Fail: model loading failed. ]")
    return dump['config']


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def label_smooth(labels, smoothing_rate, num_classes):
    """
    :param labels: before one hot
    :param smoothing_rate:
    :param num_classes:
    :return:
    """
    one_hot_labels = F.one_hot(labels, num_classes=num_classes).type(torch.float)
    new_labels = (1.0 - smoothing_rate) * one_hot_labels + smoothing_rate / num_classes
    return new_labels


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


def split_train(data, dev_ratio):
    data_cluster = {}
    for d in data:
        if d['relation'] not in data_cluster.keys():
            data_cluster[d['relation']] = [d]
        else:
            data_cluster[d['relation']] += [d]
    train_data, dev_data = [], []
    for d_c in data_cluster.values():
        dev_size = int(len(d_c) * dev_ratio)
        random.shuffle(d_c)
        dev_data += d_c[:dev_size]
        train_data += d_c[dev_size:]
    random.shuffle(train_data)
    random.shuffle(dev_data)
    return train_data, dev_data


def lr_warmup(init_lr, n_current_steps, n_warmup_steps, decay):
    if n_warmup_steps > 0:
        if n_current_steps <= n_warmup_steps:
            return init_lr / n_warmup_steps * n_current_steps
        else:
            return init_lr * (decay ** (n_current_steps - n_warmup_steps))
    else:
        return init_lr * (decay ** (n_current_steps-1))


def lr_warmup_transformer(init_lr, n_current_steps, n_warmup_steps):
    scale = np.min(np.power(n_current_steps, -0.5), np.power(n_warmup_steps, -1.5) * n_current_steps)
    return init_lr * scale


def CosineAnnealing(cur_lr, lr_min, cur_epoch, T_max):
    return (1 + math.cos(math.pi * cur_epoch / T_max)) / (1 + math.cos(math.pi * (cur_epoch - 1) / T_max)) * \
           (cur_lr - lr_min) + lr_min


def StepLR(cur_lr, step_size, gamma, cur_epoch):
    if (cur_epoch == 0) or (cur_epoch % step_size != 0):
        return cur_lr
    return cur_lr * gamma