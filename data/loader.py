"""
Data loader for TACRED json files.
"""

import json
import random
import torch
import numpy as np
from model.tree import head_to_tree, tree_to_adj, tree_to_distance
from utils.dataset_control import DATASET_CONTROLLER, ENTITY_MASK
if DATASET_CONTROLLER == 'tacred':
    from utils import constant
elif DATASET_CONTROLLER == 'semeval':
    import utils.sem_constant_aggcn as constant
    # import utils.sem_constant_own as constant
else:
    raise Exception('dataset error')
from tqdm import tqdm
from model.tree import Tree, get_shortest_path

class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, batch_size, opt, vocab, evaluation=False):
        self.batch_size = batch_size
        self.opt = opt
        self.vocab = vocab
        self.eval = evaluation
        if opt.get('dataset', 'tacred') == 'tacred':
            self.label2id = constant.LABEL_TO_ID
        else:
            self.label2id = constant.LABEL_TO_ID
            # self.label2id_no_d = constant.LABEL_TO_ID_NO_D
        self.min_sub_pos, self.max_sub_pos, \
        self.min_obj_pos, self.max_obj_pos = constant.INFINITY_NUMBER, -1, \
                                             constant.INFINITY_NUMBER, -1
        with open(filename) as infile:
            data = json.load(infile)
        self.raw_data = data
        self.max_length = 0
        data = self.preprocess(data, vocab, opt)

        # shuffle for training
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        self.id2label = dict([(v,k) for k,v in self.label2id.items()])
        # if opt.get('dataset', 'tacred') == 'tacred':
        self.labels = [self.id2label[d[-1]] for d in data]
        # else:
        #     self.labels = [self.id2label[d[-1][0]] for d in data]
        self.num_examples = len(data)

        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        print("{} batches created for {}".format(len(data), filename))

    @staticmethod
    def head_to_tree(head):
        """
        Convert a sequence of head indexes into a tree object.
        """
        root = None
        nodes = [Tree() for _ in head]

        for i in range(len(nodes)):
            h = int(head[i])
            nodes[i].idx = i
            nodes[i].dist = -1  # just a filler
            if h == 0:
                root = nodes[i]
            else:
                nodes[h - 1].add_child(nodes[i])
        assert root is not None
        return root

    def get_entity_distance(self):
        return [min(abs(data['subj_end'] - data['obj_start']), abs(data['obj_end'] - data['subj_start']))
                for data in self.raw_data]

    def get_entity_distance_tree(self):
        all_len = []
        for data in self.raw_data:
            tree = self.head_to_tree(data['stanford_head'])
            min_dist = 1e10
            for subj_idx in range(data['subj_start'], data['subj_end'] + 1):
                for obj_idx in range(data['obj_start'], data['obj_end'] + 1):
                    cur_len = get_shortest_path(tree, subj_idx, obj_idx)
                    if isinstance(cur_len, tuple):
                        cur_len = 1e10
                    if cur_len < min_dist:
                        min_dist = cur_len
            all_len.append(min_dist)

        return all_len

    def preprocess(self, data, vocab, opt):
        """ Preprocess the data and convert to ids. """
        processed = []

        for d in tqdm(data):
            tokens = list(d['token'])
            if len(list(d['token'])) < 10:
                a=1
            if ' '.join(list(d['token'])) == 'Japan \'s Konica Minolta reports record earnings':
                a=1
            if opt['lower']:
                tokens = [t.lower() for t in tokens]
            # anonymize tokens
            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']
            if ENTITY_MASK:
                tokens[ss:se+1] = ['SUBJ-'+d['subj_type']] * (se-ss+1)
                tokens[os:oe+1] = ['OBJ-'+d['obj_type']] * (oe-os+1)
            tokens = map_to_ids(tokens, vocab.word2id)
            pos = map_to_ids(d['stanford_pos'], constant.POS_TO_ID)
            try:
                ner = map_to_ids(d['stanford_ner'], constant.NER_TO_ID)
            except:
                ner = map_to_ids(['O' for _ in range(len(tokens))], constant.NER_TO_ID)
            deprel = map_to_ids(d['stanford_deprel'], constant.DEPREL_TO_ID)
            head = [int(x) for x in d['stanford_head']]
            assert any([x == 0 for x in head])
            l = len(tokens)

            subj_positions = get_positions(d['subj_start'], d['subj_end'], l)
            obj_positions = get_positions(d['obj_start'], d['obj_end'], l)
            try:
                subj_type = [constant.SUBJ_NER_TO_ID.get(d['subj_type'], 1)]
                obj_type = [constant.OBJ_NER_TO_ID.get(d['obj_type'], 1)]
            except:
                subj_type = [1]
                obj_type = [1]
            relation = self.label2id[d['relation']]
            if l > self.max_length:
                self.max_length = l
            if self.min_sub_pos > subj_positions[0]:
                self.min_sub_pos = subj_positions[0]
            if self.max_sub_pos < subj_positions[-1]:
                self.max_sub_pos = subj_positions[-1]
            if self.min_obj_pos > obj_positions[0]:
                self.min_obj_pos = obj_positions[0]
            if self.max_obj_pos < obj_positions[-1]:
                self.max_obj_pos = obj_positions[-1]
            processed += [(tokens, pos, ner, deprel, head, subj_positions, obj_positions, subj_type, obj_type, relation)]
        return processed

    def gold(self):
        """ Return gold labels as a list. """
        return self.labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        assert len(batch) == 10

        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)

        # word dropout
        if not self.eval:
            words = [word_dropout(sent, self.opt['word_dropout']) for sent in batch[0]]
        else:
            words = batch[0]

        # convert to tensors
        words = get_long_tensor(words, batch_size)
        masks = torch.eq(words, 0)
        pos = get_long_tensor(batch[1], batch_size)
        ner = get_long_tensor(batch[2], batch_size)
        deprel = get_long_tensor(batch[3], batch_size)
        head = get_long_tensor(batch[4], batch_size)
        subj_positions = get_long_tensor(batch[5], batch_size)
        obj_positions = get_long_tensor(batch[6], batch_size)
        subj_type = get_long_tensor(batch[7], batch_size)
        obj_type = get_long_tensor(batch[8], batch_size)
        # if self.opt.get('dataset', 'tacred') == 'semeval':
        #     rel_no_d = [b[1] for b in batch[9]]
        #     rels = [b[0] for b in batch[9]]
        #     rels = [torch.LongTensor(rels), torch.LongTensor(rel_no_d)]
        # else:
        rels = torch.LongTensor(batch[9])
        return (words, masks, pos, ner, deprel, head, subj_positions, obj_positions, subj_type, obj_type, rels, orig_idx)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)


def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    return ids


def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    return list(range(-start_idx, 0)) + [0]*(end_idx - start_idx + 1) + \
            list(range(1, length-end_idx))


def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(constant.PAD_ID)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens


def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]


def word_dropout(tokens, dropout):
    """ Randomly dropout tokens (IDs) and replace them with <UNK> tokens. """
    return [constant.UNK_ID if x != constant.UNK_ID and np.random.random() < dropout \
            else x for x in tokens]

