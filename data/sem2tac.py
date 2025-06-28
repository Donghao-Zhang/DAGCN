import json
import os
from stanfordcorenlp import StanfordCoreNLP
import re
from tqdm import tqdm
from collections import Counter
import copy
import numpy as np


def entity_pos(sent_list, ent_mention_token_list):
    num_mention_token = len(ent_mention_token_list)
    for token_idx in range(len(sent_list) - num_mention_token):
        if sent_list[token_idx: token_idx+num_mention_token] == ent_mention_token_list:
            return (token_idx, token_idx+num_mention_token-1, ent_mention_token_list)
    raise Exception("entity can not be found in the sentence")


def get_ent_type(e_pos, ner):
    e_ner = ner[e_pos[0]:e_pos[1]+1]
    e_types = [item_ner[1] for item_ner in e_ner]
    e_types = Counter(e_types)
    e_types = sorted(e_types.items(), key=lambda x: x[1], reverse=True)
    if e_types[0][0] == 'O':
        e_types.pop(0)
    if len(e_types) == 0:
        return 'OTHER'
    return e_types[0][0]


def get_dep_list(dep):
    num_token = len(dep)
    dep_rel = ['' for _ in range(num_token)]
    head = [-1 for _ in range(num_token)]
    for arc in dep:
        head[arc[2]-1] = arc[1]
        dep_rel[arc[2]-1] = arc[0]
    return list(zip(*[head, dep_rel]))


def semeval_process_all_class(id, sentence, rel, nlp):
    e1_mention = re.findall('<e1>(.*)</e1>', sentence)
    e2_mention = re.findall('<e2>(.*)</e2>', sentence)
    assert len(e1_mention) == 1 and len(e2_mention) == 1, \
        "sentence: %s \n has more than 1 subject or object entity" % sentence
    sent_rm_ent_mark = ' '.join(re.split('<e1>|</e1>|<e2>|</e2>', sentence))
    token = nlp.word_tokenize(sent_rm_ent_mark)
    e1_pos = entity_pos(token, nlp.word_tokenize(e1_mention[0]))
    e2_pos = entity_pos(token, nlp.word_tokenize(e2_mention[0]))
    if e2_pos[1] < e1_pos[0]:
        e1_pos, e2_pos = e2_pos, e1_pos
    pos = nlp.pos_tag(sent_rm_ent_mark)
    ner = nlp.ner(sent_rm_ent_mark)
    dep = nlp.dependency_parse(sent_rm_ent_mark)
    dep_list = get_dep_list(dep)
    e1_type = get_ent_type(e1_pos, ner)
    e2_type = get_ent_type(e2_pos, ner)
    sub, obj = e1_pos, e2_pos
    sub_type, obj_type = e1_type, e2_type
    revised_ner = copy.deepcopy(ner)
    relation_no_d = re.split('\(', rel)[0]
    for token_idx, ner_item in enumerate(revised_ner):
        if sub[0] <= token_idx <= sub[1]:
            revised_ner[token_idx] = (ner_item[0], sub_type)
        elif obj[0] <= token_idx <= obj[1]:
            revised_ner[token_idx] = (ner_item[0], obj_type)
        else:
            revised_ner[token_idx] = (ner_item[0], 'O')
    semeval_data = {
        'id': id,
        'docid': id,
        'relation': rel,
        'relation_no_d': relation_no_d,
        'token': token,
        'subj_start': sub[0],
        'subj_end': sub[1],
        'obj_start': obj[0],
        'obj_end': obj[1],
        'subj_type': sub_type,
        'obj_type': obj_type,
        'stanford_pos': [pos_item[1] for pos_item in pos],
        'stanford_ner': [ner_item[1] for ner_item in revised_ner],
        'stanford_head': [dep_item[0] for dep_item in dep_list],
        'stanford_deprel': [dep_item[1] for dep_item in dep_list],
    }
    return semeval_data


def main(data_type):
    if data_type == 'train':
        input_name = 'TRAIN_FILE.TXT'
        output_name = 'revised_train_all_class.json'
    elif data_type == 'test':
        input_name = 'TEST_FILE_FULL.TXT'
        output_name = 'revised_test_class.json'
    else:
        raise ValueError("data type error %s" % data_type)
    semeval_path = os.path.join('../dataset/SemEval', input_name)
    semeval_res_path = os.path.join('../dataset/SemEval', output_name)
    nlp = StanfordCoreNLP('../standfordcorenlp/stanford-corenlp-full-2018-10-05')
    semeval_data_processed = []
    with open(semeval_path, 'r') as f:
        sample_mark = 0
        file_lines = f.readlines()
        for line in tqdm(file_lines):
            if line == '\n':
                sample_mark = 0
                temp_data = semeval_process_all_class(id, sent, relation, nlp)
                semeval_data_processed.append(temp_data)
                continue
            if sample_mark == 0:
                items = line.strip().split('\t')
                id, sent = items
                sent = sent.strip('"')
                sample_mark += 1
            elif sample_mark == 1:
                relation = line.strip()
                sample_mark += 1
            elif sample_mark == 2:
                sample_mark += 1
    nlp.close()
    with open(semeval_res_path, 'w') as outfile:
        json.dump(semeval_data_processed, outfile)
    return semeval_data_processed


if __name__ == '__main__':
    tacred_path = os.path.join('../dataset/tacred', 'train.json')
    with open(tacred_path, 'r') as infile:
        tacred_data = json.load(infile)
    data_types = ["train", 'test']
    for data_type in data_types:
        semeval_data_processed = main(data_type)
        relations = set([item['relation'] for item in semeval_data_processed])
        print('\n')
        print("*"*10, data_type, '*'*10)
        print('data number: %d' % len(semeval_data_processed))
        print('relation number: %d' % len(relations))
        print('relation:')
        for rel in list(relations):
            print(rel)
