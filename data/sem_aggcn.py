import json
import os

data_path_own = '../dataset/SemEval'
data_path_agg = '../dataset/SemEval_aggcn'

dataset_path = ['train_all.json', 'test.json']

for path in dataset_path:
    cur_path_own = os.path.join(data_path_own, path)
    own_data = json.load(open(cur_path_own, 'r'))
    cur_path_agg = os.path.join(data_path_agg, path)
    agg_data = json.load(open(cur_path_agg, 'r'))
    for idx in range(len(own_data)):
        print(idx)
        if idx == 363:
            a=1
        # assert len(own_data[idx]['token']) >= len(agg_data[idx]['token'])
        agg_data[idx]['stanford_ner'] = own_data[idx]['stanford_ner'][:len(agg_data[idx]['token'])]
        to_add = len(agg_data[idx]['token']) - len(own_data[idx]['token'])
        agg_data[idx]['stanford_ner'] += ['O'] * to_add
        agg_data[idx]['stanford_head'] = [int(h) for h in agg_data[idx]['stanford_head']]
    cue_save_path = os.path.join(data_path_agg, 'conv' + path)
    json.dump(agg_data, open(cue_save_path, 'w'))
a=1