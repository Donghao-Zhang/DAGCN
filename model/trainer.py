"""
A trainer class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import model.multi_interaction as multi_interaction
from utils import torch_utils


class Trainer(object):
    def __init__(self, opt, emb_matrix=None):
        raise NotImplementedError

    def update(self, batch, epoch=-1):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

    def save(self, filename, epoch):
        params = {
                'model': self.model.state_dict(),
                'config': self.opt,
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")


def unpack_batch(batch, cuda):
    if cuda:
        inputs = [Variable(b.cuda()) for b in batch[:10]]
        # if isinstance(batch[10][0], list):
        #     labels = [Variable(l) for l in batch[10]]
        # else:
        labels = Variable(batch[10].cuda())
    else:
        inputs = [Variable(b) for b in batch[:10]]
        if isinstance(batch[10][0], list):
            labels = [Variable(l) for l in batch[10]]
        else:
            labels = Variable(batch[10])
    tokens = batch[0]
    head = batch[5]
    subj_pos = batch[6]
    obj_pos = batch[7]
    lens = batch[1].eq(0).long().sum(1).squeeze()
    return inputs, labels, tokens, head, subj_pos, obj_pos, lens

class GCNTrainer(Trainer):
    def __init__(self, opt, emb_matrix=None):
        self.opt = opt
        self.emb_matrix = emb_matrix
        self.model = multi_interaction.GCNClassifier(opt, emb_matrix=emb_matrix)
        # self.model = GCNClassifier(opt, emb_matrix=emb_matrix)
        self.criterion = self.create_criterion()
        self.param_num = torch_utils.get_parameter_number(self.model)
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])
        self.optimizer.zero_grad()

    def create_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def update(self, batch, epoch=-1, update_param=True):
        """
        :param batch: (words, masks, pos, ner, deprel, head, subj_positions,
        obj_positions, subj_type, obj_type, rels, orig_idx)

        inputs: (words, masks, pos, ner, deprel, head, subj_positions,
        obj_positions, subj_type, obj_type)

        tokens: words
        """
        inputs, labels, tokens, head, subj_pos, obj_pos, lens = unpack_batch(batch, self.opt['cuda'])
        # step forward
        self.model.train()
        logits, pooling_output = self.model(inputs, epoch)
        loss = self.criterion(logits, labels)
        # l2 decay on all conv layers
        if self.opt.get('conv_l2', 0) > 0:
            loss += self.model.conv_l2() * self.opt['conv_l2']
        # l2 penalty on output representations
        if self.opt.get('pooling_l2', 0) > 0:
            loss += self.opt['pooling_l2'] * (pooling_output ** 2).sum(1).mean()
        loss_val = loss.item()
        # backward
        if self.opt['gradient_accumulation_steps'] > 1:
            loss = loss / self.opt['gradient_accumulation_steps']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
        if update_param:
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss_val

    def predict(self, batch, unsort=True):
        inputs, labels, tokens, head, subj_pos, obj_pos, lens = unpack_batch(batch, self.opt['cuda'])
        orig_idx = batch[11]
        # forward
        self.model.eval()
        logits, _ = self.model(inputs)
        loss = self.criterion(logits, labels)
        probs = F.softmax(logits, 1).data.cpu().numpy()
        predictions = np.argmax(probs, axis=1).tolist()
        probs = probs.tolist()
        if unsort:
            _, predictions, probs = [list(t) for t in zip(*sorted(zip(orig_idx,\
                    predictions, probs)))]
        return predictions, probs, loss.item()
