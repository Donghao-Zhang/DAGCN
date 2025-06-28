"""
Prepare vocabulary and initial word vectors.
"""
import json
import pickle
import argparse
import numpy as np
from collections import Counter
import os
from utils import vocab, helper
from utils.dataset_control import ENTITY_MASK

# python3 prepare_vocab.py dataset/tacred dataset/vocab --glove_dir dataset/glove

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare vocab for relation extraction.')
    parser.add_argument('--data_dir', default='dataset/', help='TACRED directory.')
    parser.add_argument('--dataset', type=str, default='tacred', choices=['tacred', 'semeval'])
    parser.add_argument('--vocab_dir', default='dataset/vocab', help='Output vocab directory.')
    parser.add_argument('--glove_dir', default='dataset/glove', help='GloVe directory.')
    parser.add_argument('--wv_file', default='glove.840B.300d.txt', help='GloVe vector file.')
    parser.add_argument('--wv_dim', type=int, default=300, help='GloVe vector dimension.')
    parser.add_argument('--min_freq', type=int, default=0, help='If > 0, use min_freq as the cutoff.')
    parser.add_argument('--lower', action='store_true', help='If specified, lowercase all words.')
    
    args = parser.parse_args()
    return args


def main(args):
    # input files
    if args.dataset == 'tacred':
        data_dir = os.path.join(args.data_dir, 'tacred')
        train_file = os.path.join(data_dir, 'train.json')
        dev_file = os.path.join(data_dir, 'dev.json')
        test_file = os.path.join(data_dir, 'test.json')
    elif args.dataset == 'semeval':
        data_dir = os.path.join(args.data_dir, 'SemEval_aggcn')
        train_file = os.path.join(data_dir, 'train_all.json')
        dev_file = None
        test_file = os.path.join(data_dir, 'test.json')
    else:
        raise Exception('dataset error')

    wv_file = args.glove_dir + '/' + args.wv_file
    wv_dim = args.wv_dim

    # output files
    helper.ensure_dir(args.vocab_dir)
    if args.dataset == 'tacred':
        vocab_file = args.vocab_dir + '/vocab.pkl'
        emb_file = args.vocab_dir + '/embedding.npy'
    elif args.dataset == 'semeval':
        if not ENTITY_MASK:
            vocab_file = args.vocab_dir + '/vocab_aggcn.pkl'
            emb_file = args.vocab_dir + '/embedding_aggcn.npy'
        else:
            vocab_file = args.vocab_dir + '/vocab_aggcn_mask.pkl'
            emb_file = args.vocab_dir + '/embedding_aggcn_mask.npy'
    else:
        raise Exception('dataset error')
    # load files
    print("loading files...")
    print("ENTITY_MASK: ", ENTITY_MASK)
    train_tokens = load_tokens(train_file)
    dev_tokens = load_tokens(dev_file) if dev_file is not None else None
    test_tokens = load_tokens(test_file)
    if args.lower:
        if dev_file is not None:
            train_tokens, dev_tokens, test_tokens = [[t.lower() for t in tokens] for tokens in \
                                                     (train_tokens, dev_tokens, test_tokens)]
        else:
            train_tokens, test_tokens = [[t.lower() for t in tokens] for tokens in \
                                         (train_tokens, test_tokens)]
            dev_tokens = None

    # load glove
    print("loading glove...")
    glove_vocab = vocab.load_glove_vocab(wv_file, wv_dim)
    print("{} words loaded from glove.".format(len(glove_vocab)))
    
    print("building vocab...")
    v = build_vocab(train_tokens, glove_vocab, args.min_freq)

    print("calculating oov...")
    datasets = {'train': train_tokens, 'dev': dev_tokens, 'test': test_tokens}
    for dname, d in datasets.items():
        if d is not None:
            total, oov = count_oov(d, v)
            print("{} oov: {}/{} ({:.2f}%)".format(dname, oov, total, oov*100.0/total))
    
    print("building embeddings...")
    embedding = vocab.build_embedding(wv_file, v, wv_dim)
    print("embedding size: {} x {}".format(*embedding.shape))

    print("dumping to files...")
    with open(vocab_file, 'wb') as outfile:
        pickle.dump(v, outfile)
    np.save(emb_file, embedding)
    print("all done.")

def load_tokens(filename):
    with open(filename) as infile:
        data = json.load(infile)
        tokens = []
        for d in data:
            ts = d['token']
            ss, se, os, oe = d['subj_start'], d['subj_end'], d['obj_start'], d['obj_end']
            # do not create vocab for entity words
            if ENTITY_MASK:
                ts[ss:se+1] = ['<PAD>']*(se-ss+1)
                ts[os:oe+1] = ['<PAD>']*(oe-os+1)
            tokens += list(filter(lambda t: t!='<PAD>', ts))
    print("{} tokens from {} examples loaded from {}.".format(len(tokens), len(data), filename))
    return tokens

def build_vocab(tokens, glove_vocab, min_freq):
    """ build vocab from tokens and glove words. """
    counter = Counter(t for t in tokens)
    # if min_freq > 0, use min_freq, otherwise keep all glove words
    if min_freq > 0:
        v = sorted([t for t in counter if counter.get(t) >= min_freq], key=counter.get, reverse=True)
    else:
        v = sorted([t for t in counter if t in glove_vocab], key=counter.get, reverse=True)
    # add special tokens and entity mask tokens
    if ENTITY_MASK:
        v = constant.VOCAB_PREFIX + entity_masks() + v
    else:
        v = constant.VOCAB_PREFIX + v
    print("vocab built with {}/{} words.".format(len(v), len(counter)))
    return v

def count_oov(tokens, vocab):
    c = Counter(t for t in tokens)
    total = sum(c.values())
    matched = sum(c[t] for t in vocab)
    return total, total-matched

def entity_masks():
    """ Get all entity mask tokens as a list. """
    masks = []
    subj_entities = list(constant.SUBJ_NER_TO_ID.keys())[2:]
    obj_entities = list(constant.OBJ_NER_TO_ID.keys())[2:]
    masks += ["SUBJ-" + e for e in subj_entities]
    masks += ["OBJ-" + e for e in obj_entities]
    return masks

if __name__ == '__main__':
    args = parse_args()
    if args.dataset == 'tacred':
        from utils import constant
    elif args.dataset == 'semeval':
        import utils.sem_constant_own as constant
    else:
        raise Exception('dataset error')
    main(args)


