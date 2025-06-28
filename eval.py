"""
Run evaluation with saved models.
"""
import random
import argparse
from tqdm import tqdm
import torch

from data.loader import DataLoader
from model.trainer import GCNTrainer
from utils import torch_utils, scorer, helper
from utils.vocab import Vocab
import os
import pickle
from utils.dataset_control import DATASET_CONTROLLER
if DATASET_CONTROLLER == 'tacred':
    from utils import constant
    from utils import scorer
elif DATASET_CONTROLLER == 'semeval':
    import utils.sem_constant_aggcn as constant
    # import utils.scorer_semeval as scorer
    from utils.score_official import get_marco_f1
    from utils import scorer
else:
    raise Exception('dataset error')

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('model_dir', type=str, help='Directory of the model.')
parser.add_argument('--model', type=str, default='best_model.pt', help='Name of the model file.')
# checkpoint_epoch_100.pt
# best_model.pt
parser.add_argument('--data_dir', type=str, default='dataset')
# parser.add_argument('--dataset', type=str, default='revised_test_no_d', help="Evaluate on dev or test.")
parser.add_argument('--dataset', type=str, default='test', help="Evaluate on dev or test.")
parser.add_argument('--dataset_control', type=str, default='tacred', choices=['tacred', 'semeval'])
parser.add_argument('--model_type', type=str, default='multi_interaction')  # aggcn / multi_interaction

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true')
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(1234)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

# load opt
model_file = args.model_dir + '/' + args.model
print("Loading model from {}".format(model_file))
opt = torch_utils.load_config(model_file)
if not opt.get('model_type', False):
    opt['model_type'] = 'aggcn'
# load data
if args.dataset_control == 'tacred':
    opt['data_dir'] = os.path.join(args.data_dir, 'tacred')
    # vocab_file = args.model_dir + '/vocab.pkl'
elif args.dataset_control == 'semeval':
    # opt['data_dir'] = os.path.join(args.data_dir, 'SemEval')
    opt['data_dir'] = os.path.join(args.data_dir, 'SemEval_aggcn')
    # vocab_file = args.model_dir + '/vocab_semeval.pkl'
else:
    raise Exception('dataset error')

trainer = GCNTrainer(opt)
trainer.load(model_file)

# load vocab
vocab_file = args.model_dir + '/vocab.pkl'
vocab = Vocab(vocab_file, load=True)
assert opt['vocab_size'] == vocab.size, "Vocab size must match that in the saved model."

# load data
data_file = os.path.join(opt['data_dir'], '{}.json'.format(args.dataset))
print("Loading data from {} with batch size {}...".format(data_file, opt['batch_size']))
batch = DataLoader(data_file, opt['batch_size'], opt, vocab, evaluation=True)

helper.print_config(opt)
label2id = constant.LABEL_TO_ID
id2label = dict([(v,k) for k,v in label2id.items()])

predictions = []
all_probs = []
batch_iter = tqdm(batch)
entity_distance = batch.get_entity_distance()
entity_distance_tree = batch.get_entity_distance_tree()
for i, b in enumerate(batch_iter):
    if i == 7:
        a=1
    preds, probs, _ = trainer.predict(b)
    predictions += preds
    all_probs += probs

predictions = [id2label[p] for p in predictions]

def save_res(distance, tree_dist, gold, pred):
    assert len(distance) == len(tree_dist) == len(gold) == len(pred)
    num_sample = len(distance)
    res = [[distance[i], gold[i], pred[i]] for i in range(num_sample)]
    model_id = args.model_dir.split('/')[-1]
    pickle.dump(res, open('./res_entity_dist_{}.pkl'.format(model_id), 'wb'))
    # pickle.dump(res, open('./res_entity_dist_aggcn_my.pkl', 'wb'))
    # pickle.dump(res, open('./res_entity_dist_gcn.pkl', 'wb'))
    res_tree = [[tree_dist[i], gold[i], pred[i]] for i in range(num_sample)]
    pickle.dump(res_tree, open('./res_entity_dist_tree_{}.pkl'.format(model_id), 'wb'))
    # pickle.dump(res_tree, open('./res_entity_dist_aggcn_my_tree.pkl', 'wb'))
    # pickle.dump(res_tree, open('./res_entity_dist_gcn_tree.pkl', 'wb'))


# save_res(entity_distance, entity_distance_tree, batch.gold(), predictions)

if args.dataset_control == 'tacred':
    p, r, f1 = scorer.score(batch.gold(), predictions, verbose=True)

elif args.dataset_control == 'semeval':
    f1 = get_marco_f1(predictions, batch.gold(), gpu_idx='eval')
    p, r = 0, 0
    # p, r, f1 = scorer.score(batch.gold(), predictions, verbose=True, micro=False)
    # p, r, f1 = scorer.score_aggcn(batch.gold(), predictions, verbose=True)

else:
    raise Exception('dataset error')
print("{} set evaluate result: {:.2f}\t{:.2f}\t{:.2f}".format(args.dataset,p,r,f1))

print("Evaluation ended.")

