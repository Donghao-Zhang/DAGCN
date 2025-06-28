"""
Train a model on TACRED.
"""

import os
import sys
from datetime import datetime
import time
import numpy as np
import random
import argparse
from shutil import copyfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from data.loader import DataLoader
from model.trainer import GCNTrainer
# from utils import torch_utils, scorer, helper
from utils import torch_utils, helper
from utils.vocab import Vocab
import json
from utils.dataset_control import DATASET_CONTROLLER, ENTITY_MASK
if DATASET_CONTROLLER == 'tacred':
    from utils import constant
    from utils import scorer
elif DATASET_CONTROLLER == 'semeval':
    # import utils.sem_constant_own as constant
    import utils.sem_constant_aggcn as constant
    # import utils.scorer_semeval as scorer
    from utils.score_official import get_marco_f1
    from utils import scorer
else:
    raise Exception('dataset error')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='tacred', choices=['tacred', 'semeval'])
parser.add_argument('--data_split_ratio', type=float, default=0.1, help='train and dev data split ratio')  # 0.1875
parser.add_argument('--data_dir', type=str, default='dataset')
parser.add_argument('--vocab_dir', type=str, default='dataset/vocab')
parser.add_argument('--emb_dim', type=int, default=300, help='Word embedding dimension.')
parser.add_argument('--ner_dim', type=int, default=30, help='NER embedding dimension.')
parser.add_argument('--pos_dim', type=int, default=30, help='POS embedding dimension.')
parser.add_argument('--deprel_dim', type=int, default=120, help='dependency relation embedding dimension.')
parser.add_argument('--sub_pos_dim', type=int, default=0, help='dependency relation embedding dimension.')
parser.add_argument('--obj_pos_dim', type=int, default=0, help='dependency relation embedding dimension.')
parser.add_argument('--position_length', type=int, default=250)
parser.add_argument('--min_position', type=int, default=125)
parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
parser.add_argument('--num_inter_times', type=int, default=1)
parser.add_argument('--init_gcn_layer', type=int, default=1)
parser.add_argument('--init_rnn_layer', type=int, default=1)
parser.add_argument('--gat_heads', type=int, default=1)
parser.add_argument('--inter_times', type=int, default=1)
parser.add_argument('--hard_rnn_inter_rnn', type=str, default='rnn')
parser.add_argument('--lr_decay_change_epoch', type=int, default=200)
parser.add_argument('--lr_decay_change_ratio', type=float, default=1.)
parser.add_argument('--ctx_use_gcn', action='store_true', default=False)


parser.add_argument('--hidden_dim', type=int, default=300, help='RNN hidden state size.')
parser.add_argument('--num_init_layers', type=int, default=1, help='Num of AGGCN blocks.')  # 1
parser.add_argument('--num_layers', type=int, default=3, help='Num of AGGCN blocks.')  # 2
parser.add_argument('--input_dropout', type=float, default=0.5, help='Input dropout rate.')
parser.add_argument('--gcn_dropout', type=float, default=0.5, help='AGGCN layer dropout rate.')
parser.add_argument('--word_dropout', type=float, default=0.04, help='The rate at which randomly set a word to UNK.')
parser.add_argument('--topn', type=int, default=1e10, help='Only finetune top N word embeddings.')
parser.add_argument('--finetune_epoch', type=int, default=-1, help='finetune top N word embeddings after some epoch')
parser.add_argument('--lower', dest='lower', action='store_true', help='Lowercase all words.')
parser.add_argument('--no-lower', dest='lower', action='store_false')
parser.set_defaults(lower=False)
parser.add_argument('--weight_share', action='store_true', default=False, help='weight share among layers.')  # 整体加上不好
parser.add_argument('--norm', type=str, choices=['layer_norm', 'batch_norm', 'none_residual', 'none', 'layer_norm_no_residual',
                                                 'batch_norm_no_residual'], default='none',
                    help='norm after residual')  # layer norm 不好
parser.add_argument('--dense_connect', action='store_true', default=False, help='default not use dense connect')
parser.add_argument('--elmo_connect', action='store_true', default=False, help='default use dense connect')
parser.add_argument('--hard_iteration', action='store_true', default=False, help='default use attention connection')
parser.add_argument('--ctx_syn_update', action='store_true', default=False,
                    help='whether update syntactic in rnn and context in gcn or not')
parser.add_argument('--position_align', action='store_true', default=False,
                    help='default do not use position weight decay')
parser.add_argument('--attention', choices=['multi_head', 'dot', 'general', 'concat', 'relative_multi_head',
                                            'both_relative_multi_head', 'local_relative_multi_head',
                                            'dep_aware_multi_head', 'parameter_gen_dep_aware_multi_head',
                                            'both_relative_concat', 'syn_local_ctx_relative_multi_head',
                                            'constrain_multi_head'],
                    default='multi_head', help='context syntactic key value attention')
parser.add_argument('--dep_directed', action='store_true', default=False,
                    help='default do not consider the direction of deprel')
parser.add_argument('--merge', type=str, choices=['ali_merge', 'concat', 'highway', 'gate_merge', 'entity_gate_merge'],
                    default='concat', help='default use concat merge context and syntactic information')
parser.add_argument('--initialize', action='store_true', default=False,
                    help='use default initialization')
parser.add_argument('--initialize_method', type=str, default='none',
                    choices=['none', 'xavier_uniform_', 'xavier_normal_',
                             'kaiming_uniform_', 'kaiming_normal_', 'orthogonal_', 'uniform_', 'normal_'])
parser.add_argument('--init_rnn', action='store_true', default=False,
                    help='use ')
parser.add_argument('--prune_k', type=int, default=-1,
                    help='prune tree')
parser.add_argument('--mi_hidden', type=int, default=390, help='hidden dim of multi_interaction.')
parser.add_argument('--emb_project', action='store_true', default=False, help='hidden dim of multi_interaction.')
parser.add_argument('--syn_att_dim', type=int, default=30, help='ratio of syntactic information in attention')
parser.add_argument('--origin_gcn', action='store_true', default=False, help='use origin gcn layer.')
parser.add_argument('--gcn_layer', type=int, default=1)
parser.add_argument('--ctx_pool', type=str, default='max', choices=['max', 'avg', 'att'], help='context representation')
parser.add_argument('--sub_obj_pool', type=str, default='max', choices=['max', 'avg'], help='subject and object representation')
parser.add_argument('--self_loop', action='store_true', default=False, help='generate adj use self-loop.')
parser.add_argument('--ctx_mask_modify', action='store_true', default=False)
parser.add_argument('--ctx_syn_att', action='store_true', default=False)
parser.add_argument('--smooth_rate', type=float, default=0.1, help='deprel label smooth rate')
parser.add_argument('--meta_alpha', type=float, default=0.5, help='deprel label smooth rate')
parser.add_argument('--init_dependent', action='store_true', default=False)
parser.add_argument('--ctx_syn_together', action='store_true', default=False)
parser.add_argument('--ctx_adj', action='store_true', default=False)
parser.add_argument('--ctx_syn_adj_reverse', action='store_true', default=False)
parser.add_argument('--highway_layers', type=int, default=1)

parser.add_argument('--heads', type=int, default=3, help='Num of heads in multi-head attention.')
parser.add_argument('--sublayer_first', type=int, default=2, help='Num of the first sublayers in dcgcn block.')
parser.add_argument('--sublayer_second', type=int, default=4, help='Num of the second sublayers in dcgcn block.')

parser.add_argument('--pooling', choices=['max', 'avg', 'sum'], default='max', help='Pooling function type. Default max.')
parser.add_argument('--pooling_l2', type=float, default=0.002, help='L2-penalty for all pooling output.')
parser.add_argument('--mlp_layers', type=int, default=1, help='Number of output mlp layers.')
parser.add_argument('--no_adj', dest='no_adj', action='store_true', help="Zero out adjacency matrix for ablation.")

parser.add_argument('--no-rnn', dest='rnn', action='store_false', help='Do not use RNN layer.')
parser.add_argument('--rnn_hidden', type=int, default=300, help='RNN hidden state size.')
parser.add_argument('--rnn_layers', type=int, default=1, help='Number of RNN layers.')
parser.add_argument('--rnn_dropout', type=float, default=0.5, help='RNN dropout rate.')

parser.add_argument('--lr', type=float, default=0.7, help='Applies to sgd and adagrad.')
parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate decay rate.')
parser.add_argument('--decay_epoch', type=int, default=5, help='Decay learning rate after this epooling_l2poch.')
parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax', 'bertAdam'], default='sgd',
                    help='Optimizer: sgd, adagrad, adam or adamax.')
parser.add_argument('--num_epoch', type=int, default=100, help='Number of total training epochs.')
parser.add_argument('--batch_size', type=int, default=50, help='Training batch size.')
parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
parser.add_argument('--log_step', type=int, default=50, help='Print log every k steps.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument('--save_epoch', type=int, default=100, help='Save model checkpoints every k epochs.')
parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
parser.add_argument('--id', type=str, default='03', help='Model ID under which to save models.')
parser.add_argument('--info', type=str, default='', help='Optional info for the experiment.')

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')

parser.add_argument('--model_type', choices=['multi_interaction', 'aggcn', 'mix', 'aggcn_mi', 'gcn', 'inter_attention', 'hard_rnn'],
                    type=str, default='multi_interaction')  # aggcn / multi_interaction
parser.add_argument('--load', dest='load', action='store_true', help='Load pretrained model.')
parser.add_argument('--model_file', type=str, help='Filename of the pretrained model.')
parser.add_argument('--gpu', type=str, default='1')

args = parser.parse_args()

torch.cuda.set_device(int(args.gpu))
os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % args.gpu

if args.seed > -10:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # random.seed(1234)
    if args.cpu:
        args.cuda = False
    elif args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
random.seed(1234)
init_time = time.time()

# make opt
opt = vars(args)
label2id = constant.LABEL_TO_ID
opt['num_class'] = len(label2id)

# load vocab
if opt.get('dataset', 'tacred') == 'tacred':
    vocab_name = 'vocab.pkl'
    emb_name = 'embedding.npy'
elif opt.get('dataset', 'tacred') == 'semeval':
    if not ENTITY_MASK:
        # aggcn
        vocab_name = 'vocab_aggcn.pkl'
        emb_name = 'embedding_aggcn.npy'
    else:
        # aggcn
        vocab_name = 'vocab_aggcn_mask.pkl'
        emb_name = 'embedding_aggcn_mask.npy'
else:
    raise Exception('dataset error')

# vocab_file = opt['vocab_dir'] + '/vocab.pkl'
vocab_file = os.path.join(opt['vocab_dir'], vocab_name)
vocab = Vocab(vocab_file, load=True)
opt['vocab_size'] = vocab.size
# emb_file = opt['vocab_dir'] + '/embedding.npy'
emb_file = os.path.join(opt['vocab_dir'], emb_name)
emb_matrix = np.load(emb_file)
assert emb_matrix.shape[0] == vocab.size
assert emb_matrix.shape[1] == opt['emb_dim']

# load data
if opt.get('dataset', 'tacred') == 'tacred':
    opt['data_dir'] = os.path.join(opt['data_dir'], 'tacred')
elif opt.get('dataset', 'tacred') == 'semeval':
    # aggcn
    # opt['data_dir'] = os.path.join(opt['data_dir'], 'SemEval')
    opt['data_dir'] = os.path.join(opt['data_dir'], 'SemEval_aggcn')
    if opt.get('data_split_ratio', 0.1) != 0:
        # sem_train_path = os.path.join(opt['data_dir'], 'revised_train_all_no_d.json')
        # sem_train_path = os.path.join(opt['data_dir'], 'revised_train_all_class.json')

        sem_train_path = os.path.join(opt['data_dir'], 'train_all.json')
        sem_train_save_path = os.path.join(opt['data_dir'], 'train.json')
        sem_dev_save_path = os.path.join(opt['data_dir'], 'dev.json')
        with open(sem_train_path, 'r') as sem_file:
            train_data = json.load(sem_file)
            train, dev = torch_utils.split_train(train_data, opt.get('data_split_ratio', 0.1))
            with open(sem_train_save_path, 'w') as outfile:
                json.dump(train, outfile)
            with open(sem_dev_save_path, 'w') as outfile:
                json.dump(dev, outfile)

print("Loading data from {} with batch size {}...".format(opt['data_dir'], opt['batch_size']))
if not (opt.get('dataset', 'tacred') == 'semeval' and opt.get('data_split_ratio', 0.1) == 0):
    train_batch = DataLoader(opt['data_dir'] + '/train.json', opt['batch_size'], opt, vocab, evaluation=False)
    dev_batch = DataLoader(opt['data_dir'] + '/dev.json', opt['batch_size'], opt, vocab, evaluation=True)
    opt['dev_batch_number'] = dev_batch.num_examples
    print("dev min subject entity position: {}".format(dev_batch.min_sub_pos))
    print("dev min object entity position: {}".format(dev_batch.min_obj_pos))
else:
    train_batch = DataLoader(opt['data_dir'] + '/revised_train_all_no_d.json', opt['batch_size'], opt, vocab, evaluation=False)
    dev_batch = None
    opt['dev_batch_number'] = 0
opt['train_batch_number'] = train_batch.num_examples
print("train min subject entity position: {}".format(train_batch.min_sub_pos))
print("train min object entity position: {}".format(train_batch.min_obj_pos))
print("train max subject entity position: {}".format(train_batch.max_sub_pos))
print("train max object entity position: {}".format(train_batch.max_obj_pos))
print('*'*30)

print("train max subject entity position: {}".format(train_batch.max_sub_pos))
print("train max object entity position: {}".format(train_batch.max_obj_pos))

max_steps = len(train_batch) * opt['num_epoch']
args.total_steps = max_steps

model_id = opt['id'] if len(opt['id']) > 1 else '0' + opt['id']
model_save_dir = opt['save_dir'] + '/' + model_id
opt['model_save_dir'] = model_save_dir
helper.ensure_dir(model_save_dir, verbose=True)

# save config
helper.save_config(opt, model_save_dir + '/config.json', verbose=True)
vocab.save(model_save_dir + '/vocab.pkl')
file_logger = helper.FileLogger(model_save_dir + '/' + opt['log'], header="# epoch\ttrain_loss\tdev_loss\tdev_score\tbest_dev_score")

# print model info
helper.print_config(opt)

print('initialize model')
# model
if not opt['load']:
    trainer = GCNTrainer(opt, emb_matrix=emb_matrix)
else:
    # load pretrained model
    model_file = opt['model_file']
    print("Loading model from {}".format(model_file))
    model_opt = torch_utils.load_config(model_file)
    model_opt['optim'] = opt['optim']
    trainer = GCNTrainer(model_opt)
    trainer.load(model_file)
print("model total parameters: {}".format(trainer.param_num))
file_logger.log("model total parameters: {}".format(trainer.param_num))
id2label = dict([(v,k) for k,v in label2id.items()])
dev_score_history = []
current_lr = opt['lr']

global_step = 0
global_start_time = time.time()
format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'
# start training
update_param = False
for epoch in range(1, opt['num_epoch']+1):
    train_loss = 0
    if 0 <= opt.get('finetune_epoch', -1) <= epoch:
        trainer.model.gcn_model.emb.weight.requires_grad = True
    for i, batch in enumerate(train_batch):

        if global_step != 0 and global_step % 2 == 0:
            a=1
        if (i+1) % opt['gradient_accumulation_steps'] == 0 or i == len(train_batch) - 1:
            update_param=True
        start_time = time.time()
        global_step += 1

        loss = trainer.update(batch, epoch, update_param)
        update_param = False
        train_loss += loss
        if global_step % opt['log_step'] == 0:
            duration = time.time() - start_time
            if opt['optim'] == 'bertAdam':
                current_lr = trainer.optimizer.get_lr()[0]
            print(format_str.format(datetime.now(), global_step, max_steps, epoch,\
                    opt['num_epoch'], loss, duration, current_lr))
        torch.cuda.empty_cache()


    # eval on dev
    if dev_batch is not None:
        print("Evaluating on dev set...")
        predictions = []
        dev_loss = 0
        for i, batch in enumerate(dev_batch):
            preds, _, loss = trainer.predict(batch)
            predictions += preds
            dev_loss += loss
        predictions = [id2label[p] for p in predictions]
        train_loss = train_loss / train_batch.num_examples * opt['batch_size'] # avg loss per batch
        dev_loss = dev_loss / dev_batch.num_examples * opt['batch_size']

        if opt.get('dataset', 'tacred') == 'tacred':
            dev_p, dev_r, dev_f1 = scorer.score(dev_batch.gold(), predictions)

        elif opt.get('dataset', 'tacred') == 'semeval':
            # dev_p, dev_r, dev_f1 = scorer.score_aggcn(dev_batch.gold(), predictions)
            #
            dev_f1 = get_marco_f1(predictions, dev_batch.gold(), gpu_idx=args.gpu)
            dev_p, dev_r = 0, 0
            # dev_p, dev_r, dev_f1 = scorer.score(dev_batch.gold(), predictions, micro=False)
        else:
            raise Exception('dataset error')
        # dev_p, dev_r, dev_f1 = scorer.score(dev_batch.gold(), predictions)
        print("epoch {}: train_loss = {:.6f}, dev_loss = {:.6f}, dev_f1 = {:.4f}".
              format(epoch, train_loss, dev_loss, dev_f1))
        dev_score = dev_f1
        file_logger.log("{}\t{:.6f}\t{:.6f}\t{:.4f}\t{:.4f}".format(epoch, train_loss, dev_loss, dev_score, max([dev_score] + dev_score_history)))
        # save
        model_file = model_save_dir + '/checkpoint_epoch_{}.pt'.format(epoch)
        trainer.save(model_file, epoch)
        if epoch == 1 or dev_score > max(dev_score_history):
            copyfile(model_file, model_save_dir + '/best_model.pt')
            print("new best model saved.")
            file_logger.log("new best model saved at epoch {}: {:.2f}\t{:.2f}\t{:.2f}".
                            format(epoch, dev_p * 100, dev_r * 100, dev_score * 100))
        if epoch % opt['save_epoch'] != 0:
            os.remove(model_file)

        # lr schedule

        # current_lr *= opt['lr_decay']
        # trainer.update_lr(current_lr)

        decay_epoch_number = 1
        decay_status = True
        for s in dev_score_history[-decay_epoch_number:]:
            decay_status = decay_status and dev_score <= s
        # if len(dev_score_history) > opt['decay_epoch'] and dev_score <= dev_score_history[-1] and \
        #         opt['optim'] in ['sgd', 'adagrad', 'adadelta']:
        if epoch < opt.get('lr_decay_change_epoch', 200):
            lr_decay = opt['lr_decay']
        else:
            lr_decay = opt['lr_decay'] * opt.get('lr_decay_change_ratio', 1.0) \
                if opt['lr_decay'] * opt.get('lr_decay_change_ratio', 1.0) < 1 else 1
        if len(dev_score_history) > opt['decay_epoch'] and decay_status and \
                opt['optim'] in ['sgd', 'adagrad', 'adadelta']:
            current_lr *= lr_decay
            trainer.update_lr(current_lr)

        dev_score_history += [dev_score]
        print("")
    if epoch == opt['num_epoch']:
        model_file = model_save_dir + '/checkpoint_epoch_{}.pt'.format(epoch)
        trainer.save(model_file, epoch)
        file_logger.log('save model at epoch {}'.format(epoch))

print("Training ended with {} epochs.".format(epoch))

