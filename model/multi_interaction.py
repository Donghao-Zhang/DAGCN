"""
GCN model for relation extraction.
"""
import copy
import torch.nn.functional as F
from torch.autograd import Variable
from model.attention import *
from model.nn_decorator import LSTM_decorator
from model.tree import head_to_tree, tree_to_adj
from utils import torch_utils
from utils.dataset_control import DATASET_CONTROLLER
if DATASET_CONTROLLER == 'tacred':
    from utils import constant
elif DATASET_CONTROLLER == 'semeval':
    import utils.sem_constant_own as constant
else:
    raise Exception('dataset error')


class GCNClassifier(nn.Module):
    """ A wrapper classifier for GCNRelationModel. """

    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        self.gcn_model = GCNRelationModel(opt, emb_matrix=emb_matrix)
        self.opt = opt
        self.classifier = self.output_module()

    def create_ffnn(self, out_dim):
        classifier = nn.Linear(self.opt['hidden_dim'], out_dim)
        return classifier

    def output_module(self):
        classifier = self.create_ffnn(self.opt['num_class'])
        return classifier

    def forward(self, inputs, epoch=-1):
        outputs, pooling_output = self.gcn_model(inputs, epoch)
        logits = self.classifier(outputs)
        return logits, pooling_output


class GCNRelationModel(nn.Module):
    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        self.opt = opt
        self.emb_matrix = emb_matrix
        self.epoch = 0

        # create embedding layers
        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)
        self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), opt['pos_dim'], padding_idx=constant.PAD_ID) \
            if opt['pos_dim'] > 0 else None
        self.ner_emb = nn.Embedding(len(constant.NER_TO_ID), opt['ner_dim'], padding_idx=constant.PAD_ID) \
            if opt['ner_dim'] > 0 else None
        self.deprel_emb = nn.Embedding(len(constant.DEPREL_TO_ID), opt['deprel_dim'], padding_idx=constant.PAD_ID) \
            if opt['deprel_dim'] > 0 else None
        self.sub_pos_emb = nn.Embedding(opt['position_length'], opt['sub_pos_dim']) \
            if opt['sub_pos_dim'] > 0 else None
        self.obj_pos_emb = nn.Embedding(opt['position_length'], opt['obj_pos_dim']) \
            if opt['obj_pos_dim'] > 0 else None
        self.init_embeddings()
        embeddings = (self.emb, self.pos_emb, self.ner_emb, self.deprel_emb, self.sub_pos_emb, self.obj_pos_emb)
        self.all_dim = opt['emb_dim'] + opt['pos_dim'] + opt['ner_dim'] + opt['deprel_dim']
        self.hidden_dim = opt['mi_hidden']
        if self.opt.get('emb_project', False):
            self.out_dim = self.hidden_dim + opt['deprel_dim']
        else:
            self.out_dim = self.all_dim
        # gcn layer
        self.gcn = MIGCN(opt, embeddings)

        # mlp output layer
        in_dim = (opt['emb_dim'] + opt['pos_dim'] + opt['ner_dim'] + opt['deprel_dim'] + opt['sub_pos_dim'] + opt['obj_pos_dim']) * 3
        if self.opt.get('emb_project', False):
            in_dim = (opt['mi_hidden'] + opt['deprel_dim']) * 3
        # self.drop = nn.Dropout(opt['input_dropout'])
        self.in_dim = in_dim
        self.out_mlp = self.output_module()
        # self.out_mlp_1 = self.output_module()
        # self.out_mlp_2 = self.output_module()
        if opt.get('ctx_pool', 'max') == 'att':
            self.context_attention = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim),
                nn.Tanh(),
                nn.Linear(self.out_dim, 1)
            )
        self.train_dist = []
        self.dev_dist = []

    def create_ffnn(self, in_dim):
        layers = [nn.Linear(in_dim, self.opt['hidden_dim']), nn.ReLU()]
        for _ in range(self.opt['mlp_layers'] - 1):
            layers += [nn.Linear(self.opt['hidden_dim'], self.opt['hidden_dim']), nn.ReLU()]
        out_mlp = nn.Sequential(*layers)
        return out_mlp

    def output_module(self):
        out_mlp = self.create_ffnn(self.in_dim)
        return out_mlp

    def init_embeddings(self):
        if self.opt['pos_dim'] > 0:
            self.pos_emb.weight.data[1:, :].uniform_(-1.0, 1.0)
        if self.opt['ner_dim'] > 0:
            self.ner_emb.weight.data[1:, :].uniform_(-1.0, 1.0)
        if self.opt['deprel_dim'] > 0:
            self.deprel_emb.weight.data[1:, :].uniform_(-1.0, 1.0)
        if self.opt['sub_pos_dim'] > 0:
            self.sub_pos_emb.weight.data[1:, :].uniform_(-1.0, 1.0)
        if self.opt['obj_pos_dim'] > 0:
            self.obj_pos_emb.weight.data[1:, :].uniform_(-1.0, 1.0)
        if self.emb_matrix is None:
            self.emb.weight.data[1:, :].uniform_(-1.0, 1.0)
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb.weight.data.copy_(self.emb_matrix)
        # decide finetuning
        if self.opt['topn'] <= 0:
            print("Do not finetune word embedding layer.")
            self.emb.weight.requires_grad = False
        elif self.opt['topn'] < self.opt['vocab_size']:
            print("Finetune top {} word embeddings.".format(self.opt['topn']))
            self.emb.weight.register_hook(lambda x: torch_utils.keep_partial_grad(x, self.opt['topn'],
                                                                                  self.opt.get('finetune_epoch', -1),
                                                                                  self.epoch))
        else:
            print("Finetune all embeddings.")

    def forward(self, inputs, epoch=-1):
        self.epoch = epoch
        words, masks, pos, ner, deprel, head, subj_pos, obj_pos, subj_type, obj_type = inputs # unpack
        l = (masks.data.cpu().numpy() == 0).astype(np.int64).sum(1)
        maxlen = max(l)

        def inputs_to_tree_reps(head, l, prune, subj_pos, obj_pos, directed=False, self_loop=False):
            trees = [head_to_tree(head[i], l[i], prune, subj_pos[i], obj_pos[i], i) for i in range(len(l))]
            # a = time.clock()
            adj = [tree_to_adj(maxlen, tree, directed=directed, self_loop=self_loop).reshape(1, maxlen, maxlen) for tree in trees]
            # print(time.clock()-a)
            adj = np.concatenate(adj, axis=0)
            # # self-connect, 已经在 GCN 中有
            # adj += np.repeat(np.eye(np.shape(adj)[1])[np.newaxis, :, :], np.shape(adj)[0], axis=0)
            adj = torch.from_numpy(adj)
            return Variable(adj.cuda()) if self.opt['cuda'] else Variable(adj)

        adj = inputs_to_tree_reps(head.data, l, self.opt.get('prune_k', -1), subj_pos.data, obj_pos.data,
                                  self.opt['dep_directed'], self.opt.get('self_loop', False))

        h, pool_mask = self.gcn(adj, inputs)

        # pooling
        subj_mask, obj_mask = subj_pos.eq(0).eq(0).unsqueeze(2) + masks.unsqueeze(2), \
                              obj_pos.eq(0).eq(0).unsqueeze(2) + masks.unsqueeze(2)  # invert mask
        subj_out = pool(h, subj_mask, type=self.opt.get('sub_obj_pool', 'max'))
        obj_out = pool(h, obj_mask, type=self.opt.get('sub_obj_pool', 'max'))
        if self.opt.get('ctx_mask_modify', False):
            pool_mask = (pool_mask + (-1) * (subj_mask - 1) + (-1) * (obj_mask - 1)).ne(0)
        if self.opt.get('ctx_pool', 'max') != 'att':
            h_out = pool(h, pool_mask, type=self.opt.get('ctx_pool', 'max'))
        else:
            scores = self.context_attention(h)
            scores = scores.masked_fill(pool_mask, -constant.INFINITY_NUMBER)
            p_attn = F.softmax(scores, dim=1)
            h_out = torch.bmm(h.permute(0, 2, 1), p_attn).squeeze(2)
        h_out = h_out.masked_fill_(pool_mask.eq(0).squeeze(-1).sum(-1, keepdim=True).eq(0), 0)
        outputs = torch.cat([h_out, subj_out, obj_out], dim=1)
        outputs = self.out_mlp(outputs)
        return outputs, h_out


class MIGCN(nn.Module):
    def __init__(self, opt, embeddings):
        super().__init__()
        self.opt = opt
        self.all_dim = opt['emb_dim'] + opt['pos_dim'] + opt['ner_dim'] + opt['deprel_dim'] + \
                       opt['sub_pos_dim'] + opt['obj_pos_dim']
        self.num_deprel = len(constant.DEPREL_TO_ID)
        self.hidden_dim = opt['mi_hidden']
        if self.opt.get('emb_project', False):
            self.in_dim = self.hidden_dim + opt['deprel_dim']
        else:
            self.in_dim = self.all_dim
        self.context_dim = self.in_dim - opt['syn_att_dim']
        self.systactic_dim = opt['syn_att_dim']
        if opt['ctx_syn_update']:
            self.rnn_out = self.in_dim
            self.gcn_out = self.in_dim
        else:
            self.rnn_out = self.context_dim
            self.gcn_out = self.systactic_dim
        assert self.rnn_out % 2 == 0, "rnn output is %.1f" % self.rnn_out / 2
        self.use_cuda = opt['cuda']

        self.num_layers = opt['num_layers']
        self.heads = opt['heads']
        if opt['norm'] == 'layer_norm' or opt['norm'] == 'layer_norm_no_residual':
            self.norms = nn.ModuleList([
                nn.LayerNorm(self.in_dim) for _ in range(self.num_layers * 2)
            ])
        elif opt['norm'] == 'batch_norm' or opt['norm'] == 'batch_norm_no_residual':
            self.norms = nn.ModuleList([
                nn.BatchNorm1d(self.in_dim) for _ in range(self.num_layers * 2)
            ])
        else:
            self.norms = None

        if not self.opt['weight_share']:
            self.rnns, self.gcns, self.attns = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
            if not self.opt.get('init_dependent', False):
                if self.opt.get('ctx_use_gcn', False):
                    rnn_init = [OriginGraphConvLayer(opt, self.in_dim, self.rnn_out, self.opt.get('init_rnn_layer', 1))
                                for _ in range(self.opt['num_init_layers'])]
                    rnns_inter = [OriginGraphConvLayer(opt, self.in_dim, self.rnn_out, self.opt['rnn_layers'])
                                 for _ in range(self.num_layers - self.opt['num_init_layers'])]
                else:
                    rnn_init = [LSTM_decorator(self.in_dim, int(self.rnn_out / 2), opt.get('init_rnn_layer', 1),
                                               dropout=opt['rnn_dropout']) for _ in range(self.opt['num_init_layers'])]
                    rnns_inter = [LSTM_decorator(self.in_dim, int(self.rnn_out / 2), opt['rnn_layers'],
                                                 dropout=opt['rnn_dropout'])
                                  for _ in range(self.num_layers - self.opt['num_init_layers'])]
                self.rnns = nn.ModuleList(rnn_init + rnns_inter)
            if self.opt['origin_gcn']:
                if not self.opt.get('init_dependent', False):
                    gcn_init = [OriginGraphConvLayer(opt, self.in_dim, self.gcn_out, self.opt.get('init_gcn_layer', 1))
                                for _ in range(self.opt['num_init_layers'])]
                    gcn_inter = [OriginGraphConvLayer(opt, self.in_dim, self.gcn_out, self.opt['gcn_layer'])
                                 for _ in range(self.num_layers - self.opt['num_init_layers'])]
                    self.gcns = nn.ModuleList(gcn_init + gcn_inter)
                else:
                    self.gcns = nn.ModuleList()
                    self.gcns.extend([
                        OriginGraphConvLayer(opt, self.in_dim, self.in_dim, self.opt['gcn_layer'])
                        for _ in range(self.opt['num_init_layers'])
                    ])
                    self.gcns.extend([
                        OriginGraphConvLayer(opt, self.in_dim, self.gcn_out, self.opt['gcn_layer'])
                        for _ in range(self.num_layers - self.opt['num_init_layers'])
                    ])
            else:
                if not self.opt.get('init_dependent', False):
                    gcn_init = [GraphConvLayer(opt, self.in_dim, self.gcn_out, self.opt.get('init_gcn_layer', 1))
                                for _ in range(self.opt['num_init_layers'])]
                    gcn_inter = [GraphConvLayer(opt, self.in_dim, self.gcn_out, self.opt['gcn_layer'])
                                for _ in range(self.num_layers-self.opt['num_init_layers'])]
                    self.gcns = nn.ModuleList(gcn_init+gcn_inter)
            if self.opt['attention'] != 'parameter_gen_dep_aware_multi_head':
                self.attns = self.attention_module_create(self.num_layers - self.opt['num_init_layers'])

        if not self.opt.get('ctx_syn_together', False):
            if self.opt['dense_connect']:
                self.context_aggregate_W = nn.Linear(self.num_layers * self.in_dim, self.in_dim)
                self.syntactic_aggregate_W = nn.Linear(self.num_layers * self.in_dim, self.in_dim)
            if self.opt.get('elmo_connect', False):
                self.ctx_elmo_beta = nn.Parameter(torch.zeros(self.num_layers, 1))
                self.ctx_elmo_gama = nn.Parameter(torch.ones(1))

                self.syn_elmo_beta = nn.Parameter(torch.zeros(self.num_layers, 1))
                self.syn_elmo_gama = nn.Parameter(torch.ones(1))

        if self.opt.get('merge', 'concat') == 'ali_merge':
            self.ali_gate = nn.Linear(3 * self.in_dim, 1)
            self.ctx_syn_aggregate_W = nn.Linear(3 * self.in_dim, self.in_dim)
        elif self.opt.get('merge', 'concat') == 'highway':
            highway_layers = opt.get('highway_layers', 1)
            self.highway_gate = nn.ModuleList([
                nn.Linear(2 * self.in_dim, 2 * self.in_dim) for _ in range(highway_layers)
            ])
            self.highway_transform = nn.ModuleList([
                nn.Linear(2 * self.in_dim, 2 * self.in_dim) for _ in range(highway_layers)
            ])
            self.ctx_syn_aggregate_W = nn.Linear(2 * self.in_dim, self.in_dim)
        elif self.opt.get('merge', 'concat') == 'gate_merge':
            if not self.opt.get('ctx_syn_together', False):
                self.gate = nn.Sequential(
                    nn.Linear(2 * self.in_dim, 2 * self.in_dim),
                    nn.Sigmoid()
                )
                self.ctx_syn_aggregate_W = nn.Linear(2 * self.in_dim, self.in_dim)
            else:
                self.gate = nn.Sequential(
                    nn.Linear(self.num_layers * self.in_dim * 2 - self.opt['num_init_layers'] * self.in_dim,
                              self.num_layers * self.in_dim * 2 - self.opt['num_init_layers'] * self.in_dim),
                    nn.Sigmoid()
                )
                self.ctx_syn_aggregate_W = nn.Linear(self.num_layers * self.in_dim * 2 -
                                                     self.opt['num_init_layers'] * self.in_dim,
                                                     self.in_dim)

        elif self.opt.get('merge', 'concat') == 'entity_gate_merge':
            self.gate = nn.Sequential(
                nn.Linear(4 * self.in_dim, 2 * self.in_dim),
                nn.Sigmoid()
            )
            self.ctx_syn_aggregate_W = nn.Linear(2 * self.in_dim, self.in_dim)
        else:
            if not self.opt.get('ctx_syn_together', False):
                self.ctx_syn_aggregate_W = nn.Linear(2 * self.in_dim, self.in_dim)
            else:
                self.ctx_syn_aggregate_W = nn.Linear(self.num_layers * self.in_dim * 2 -
                                                     self.opt['num_init_layers'] * self.in_dim,
                                                     self.in_dim)
        self.in_drop = nn.Dropout(opt['input_dropout'])
        if self.opt.get('init_rnn', False):
            self.init_rnn = LSTM_decorator(self.all_dim - opt['deprel_dim'],
                                           int((self.all_dim - opt['deprel_dim']) / 2),
                                           opt['rnn_layers'])
        if self.opt.get('emb_project', False):
            self.emb_project = nn.Linear(opt['emb_dim'] + opt['pos_dim'] + opt['ner_dim'], self.hidden_dim)
        if self.opt.get('initialize', False):
            self.init_model()
        self.emb, self.pos_emb, self.ner_emb, self.deprel_emb, self.sub_pos_emb, self.obj_pos_emb = embeddings

    def init_model(self):
        if self.opt.get('initialize_method', 'none') != 'none':
            init_func = getattr(nn.init, self.opt['initialize_method'])
            for p in self.parameters():
                if p.dim() > 1:
                    init_func(p)

    def sigle_att(self, q_dim, k_dim, v_dim, heads, dropout):
        if self.opt['attention'] == 'multi_head':
            attns = MultiHeadAttention(q_dim, k_dim, v_dim, heads, dropout)
        elif self.opt['attention'] == 'both_relative_multi_head':
            # attns = RelativeMultiHeadAttention(q_dim, k_dim, v_dim, heads, dropout)
            attns = MultiLayerRelativeMultiHead(q_dim, k_dim, v_dim, heads, self.opt.get('inter_times', 1), dropout)
        elif self.opt['attention'] == 'constrain_multi_head':
            attns = MultiLayerConstrainMultiHeadAttention(q_dim, k_dim, v_dim, heads, self.opt.get('inter_times', 1), dropout)
        elif self.opt['attention'] == 'both_relative_concat':
            attns = ConcatAttention(q_dim, k_dim, v_dim, dropout)
        elif self.opt['attention'] == 'syn_local_ctx_relative_multi_head':
            attns = RelativeMultiHeadAttention(q_dim, k_dim, v_dim, heads, dropout)
        elif self.opt['attention'] == 'local_relative_multi_head':
            attns = RelativeMultiHeadAttention(q_dim, k_dim, v_dim, heads, dropout)
        elif self.opt['attention'] == 'dot':
            assert q_dim == k_dim, \
                "dot attention context_dim should equal to systactic_dim"
            attns = DotAttention(q_dim, k_dim, dropout)
        elif self.opt['attention'] == 'general':
            attns = GeneralDotAttention(q_dim, k_dim, v_dim, dropout)
        elif self.opt['attention'] == 'concat':
            attns = ConcatAttention(q_dim, k_dim, v_dim, dropout)
        return attns

    def attention_module_create(self, num_layers):
        attns = nn.ModuleList()
        if not self.opt['hard_iteration']:
            if self.opt.get('ctx_syn_att', False):
                if self.opt.get('num_inter_times', 1) == 1:
                    for i in range(num_layers):
                        if self.opt['attention'] != 'local_relative_multi_head':
                            attns.append(self.sigle_att(self.context_dim, self.systactic_dim, self.systactic_dim,
                                                        self.heads, self.opt['input_dropout']))
                            attns.append(self.sigle_att(self.systactic_dim, self.context_dim, self.context_dim,
                                                        self.heads, self.opt['input_dropout']))
                        else:
                            attns.append(
                                MultiLayerRelativeMultiHead(self.context_dim, self.systactic_dim, self.systactic_dim,
                                                            self.heads, self.opt.get('inter_times', 1),
                                                            self.opt['input_dropout']))
                            attns.append(
                                LocalMultiHeadAttention(self.systactic_dim, self.context_dim, self.context_dim,
                                                        self.heads, self.opt['input_dropout']))
                else:
                    for i in range(num_layers * self.opt.get('num_inter_times', 1)):
                        attns.append(self.sigle_att(self.context_dim, self.systactic_dim, self.systactic_dim,
                                                    self.heads, self.opt['input_dropout']))
                        attns.append(self.sigle_att(self.systactic_dim, self.context_dim, self.context_dim,
                                                    self.heads, self.opt['input_dropout']))
        return attns

    def input_emb(self, words, pos, ner, deprel, sub_pos, obj_pos, masks):
        word_embs = self.emb(words)
        context_embs = [word_embs]
        syntactic_embs = [self.deprel_emb(deprel)] if self.opt['deprel_dim'] > 0 else []

        if self.opt['pos_dim'] > 0:
            context_embs += [self.pos_emb(pos)]
        if self.opt['ner_dim'] > 0:
            context_embs += [self.ner_emb(ner)]
        if self.opt['sub_pos_dim'] > 0:
            context_embs += [self.sub_pos_emb(sub_pos+self.opt['min_position'])]
        if self.opt['obj_pos_dim'] > 0:
            context_embs += [self.obj_pos_emb(obj_pos+self.opt['min_position'])]

        if self.opt.get('init_rnn', False):
            context_embs = torch.cat(context_embs, dim=2)
            context_embs = [self.context_encode(context_embs, masks, context_embs.size()[0], -1)]
        if self.opt.get('emb_project', False):
            context_embs = torch.cat(context_embs, dim=2)
            context_embs = [self.emb_project(context_embs)]
        context_inputs = context_embs + syntactic_embs
        context_inputs = torch.cat(context_inputs, dim=2)
        context_inputs = self.in_drop(context_inputs)
        syntactic_inputs = context_inputs
        return context_inputs, syntactic_inputs

    def norm_connect(self, context_input, context_embs, syntactic_input, syntactic_embs, layer_idx):
        if self.opt['norm'] == 'layer_norm':  # 尝试不加 residual
            context_output = self.norms[layer_idx * 2](context_input + context_embs)
            syntactic_output = self.norms[layer_idx * 2 + 1]((syntactic_input + syntactic_embs))
        elif self.opt['norm'] == 'layer_norm_no_residual':  # 尝试不加 residual
            context_output = self.norms[layer_idx * 2](context_input)
            syntactic_output = self.norms[layer_idx * 2 + 1](syntactic_input)
        elif self.opt['norm'] == 'batch_norm':
            context_output = self.norms[layer_idx * 2](
                (context_input + context_embs).permute(0, 2, 1)
            ).permute(0, 2, 1)
            syntactic_output = self.norms[layer_idx * 2 + 1](
                (syntactic_input + syntactic_embs).permute(0, 2, 1)
            ).permute(0, 2, 1)
        elif self.opt['norm'] == 'batch_norm_no_residual':
            context_output = self.norms[layer_idx * 2](
                context_input.permute(0, 2, 1)
            ).permute(0, 2, 1)
            syntactic_output = self.norms[layer_idx * 2 + 1](
                syntactic_input.permute(0, 2, 1)
            ).permute(0, 2, 1)
        elif self.opt['norm'] == 'none_residual':
            context_output = context_input + context_embs
            syntactic_output = syntactic_input + syntactic_embs
        else:
            context_output = context_input
            syntactic_output = syntactic_input
        return context_output, syntactic_output

    def dependent_encode(self, context_input, syntactic_input, adj, masks, layer_idx, ctx_adj=None):
        # context_syn_introduce = self.context_encode(context_input, masks, context_input.size()[0], layer_idx)
        if self.opt.get('ctx_use_gcn', False):
            context_syn_introduce = self.rnns[layer_idx](ctx_adj, context_input)
        else:
            context_syn_introduce = self.rnns[layer_idx](context_input, masks, constant.PAD_ID)
        syn_context_introduce = self.gcns[layer_idx](adj, syntactic_input)

        if self.opt['ctx_syn_update']:
            context_output = context_syn_introduce
            syntactic_output = syn_context_introduce
        else:
            context_output = torch.cat([context_syn_introduce, context_input[:, :, self.context_dim:]], dim=2)
            syntactic_output = torch.cat([syntactic_input[:, :, :self.context_dim], syn_context_introduce], dim=2)
            # context_output = context_syn_introduce
            # syntactic_output = syn_context_introduce
        return context_output, syntactic_output


    def context_syn_init_emb(self, context_embs, syntactic_embs, adj, masks, layer_idx, ctx_adj=None):
        context_input, syntactic_input = context_embs, syntactic_embs
        if not self.opt.get('init_dependent', False):
            if not self.opt['weight_share']:
                context_input, syntactic_input = \
                    self.dependent_encode(context_input, syntactic_input, adj, masks, layer_idx, ctx_adj=ctx_adj)
            else:
                context_input, syntactic_input = self.dependent_encode(context_input, syntactic_input, adj, masks, 0)
            context_input = self.in_drop(torch.cat([context_input[:, :, :self.context_dim],
                                                    syntactic_input[:, :, self.context_dim:]], dim=2))
            # context_input = self.in_drop(torch.cat([context_input, syntactic_input], dim=2))
            syntactic_input = context_input
        else:
            # context_input = self.context_encode(context_input, masks, context_input.size()[0], layer_idx)
            context_input = self.rnns[layer_idx](context_input, masks, constant.PAD_ID)

            syntactic_input = self.gcns[layer_idx](adj, syntactic_input)
            context_input = self.in_drop(context_input)
            syntactic_input = self.in_drop(syntactic_input)
        context_input, syntactic_input = \
            self.norm_connect(context_input, context_embs, syntactic_input, syntactic_embs, layer_idx)
        return context_input, syntactic_input

    def context_syn_att(self, context_input, syntactic_input, adj, dep_rel_smooth,
                        src_mask, atten_layer_idx, position_emb=None, ctx_adj=None):
        if not self.opt['hard_iteration']:
            if self.opt.get('ctx_syn_att', False):
                for i in range(self.opt.get('num_inter_times', 1)):
                    if self.opt['attention'] == 'dep_aware_multi_head':

                        context_output = torch.cat(
                            [context_input[:, :, :self.context_dim],
                             self.attns[atten_layer_idx * 2](context_input[:, :, :self.context_dim],
                                                             syntactic_input[:, :, self.context_dim:],
                                                             syntactic_input[:, :, self.context_dim:],
                                                             dep_rel_smooth, src_mask)], dim=2
                        )
                        syntactic_output = torch.cat([
                            self.attns[atten_layer_idx * 2 + 1](syntactic_input[:, :, self.context_dim:],
                                                                context_input[:, :, :self.context_dim],
                                                                context_input[:, :, :self.context_dim],
                                                                dep_rel_smooth, src_mask),
                            syntactic_input[:, :, self.context_dim:]], dim=2
                        )
                    elif self.opt['attention'] == 'parameter_gen_dep_aware_multi_head':
                        context_output = torch.cat(
                            [context_input[:, :, :self.context_dim],
                             self.attns[atten_layer_idx * 2](context_input[:, :, :self.context_dim],
                                                             syntactic_input[:, :, self.context_dim:],
                                                             syntactic_input[:, :, self.context_dim:],
                                                             src_mask, position_emb)], dim=2
                        )
                        syntactic_output = torch.cat([
                            self.attns[atten_layer_idx * 2 + 1](syntactic_input[:, :, self.context_dim:],
                                                                context_input[:, :, :self.context_dim],
                                                                context_input[:, :, :self.context_dim],
                                                                dep_rel_smooth, self.s2c_meta_weight,
                                                                src_mask, adj),
                            syntactic_input[:, :, self.context_dim:]], dim=2
                        )
                    elif self.opt['attention'] == 'relative_multi_head':
                        context_output = torch.cat(
                            [context_input[:, :, :self.context_dim],
                             syntactic_input[:, :, self.context_dim:]], dim=2
                        )
                        syntactic_output = torch.cat([
                            self.attns[atten_layer_idx](syntactic_input[:, :, self.context_dim:],
                                                        context_input[:, :, :self.context_dim],
                                                        context_input[:, :, :self.context_dim], src_mask, adj),
                            syntactic_input[:, :, self.context_dim:]], dim=2
                        )
                    elif self.opt['attention'] == 'both_relative_multi_head' or \
                            self.opt['attention'] == 'both_relative_concat' or \
                            self.opt['attention'] == 'constrain_multi_head':
                        if self.opt.get('ctx_adj', False):
                            ctx_adj = ctx_adj
                        else:
                            ctx_adj = adj
                        if not self.opt.get('ctx_syn_adj_reverse', False):
                            ctx_adj, syn_adj = ctx_adj, adj
                        else:
                            syn_adj, ctx_adj = ctx_adj, adj
                        context_output = torch.cat(
                            [context_input[:, :, :self.context_dim],
                             self.attns[atten_layer_idx * 2](context_input[:, :, :self.context_dim],
                                                             syntactic_input[:, :, self.context_dim:],
                                                             syntactic_input[:, :, self.context_dim:], src_mask, ctx_adj)], dim=2
                        )
                        syntactic_output = torch.cat([
                            self.attns[atten_layer_idx * 2 + 1](syntactic_input[:, :, self.context_dim:],
                                                                context_input[:, :, :self.context_dim],
                                                                context_input[:, :, :self.context_dim], src_mask, syn_adj),
                            syntactic_input[:, :, self.context_dim:]], dim=2
                        )
                    elif self.opt['attention'] == 'syn_local_ctx_relative_multi_head':
                        if self.opt.get('ctx_adj', False):
                            ctx_adj = ctx_adj
                        else:
                            ctx_adj = adj
                        if not self.opt.get('ctx_syn_adj_reverse', False):
                            ctx_adj, syn_adj = ctx_adj, adj
                        else:
                            syn_adj, ctx_adj = ctx_adj, adj
                        context_output = torch.cat(
                            [context_input[:, :, :self.context_dim],
                             self.attns[atten_layer_idx * 2](context_input[:, :, :self.context_dim],
                                                             syntactic_input[:, :, self.context_dim:],
                                                             syntactic_input[:, :, self.context_dim:], src_mask, ctx_adj)], dim=2
                        )
                        syntactic_output = torch.cat([
                            self.attns[atten_layer_idx * 2 + 1](syntactic_input[:, :, self.context_dim:],
                                                                context_input[:, :, :self.context_dim],
                                                                context_input[:, :, :self.context_dim], src_mask),
                            syntactic_input[:, :, self.context_dim:]], dim=2
                        )
                    elif self.opt['attention'] == 'local_relative_multi_head':
                        context_output = torch.cat(
                            [context_input[:, :, :self.context_dim],
                             self.attns[atten_layer_idx * 2](context_input[:, :, :self.context_dim],
                                                             syntactic_input[:, :, self.context_dim:],
                                                             syntactic_input[:, :, self.context_dim:], src_mask, adj)], dim=2
                        )
                        syntactic_output = torch.cat([
                            self.attns[atten_layer_idx * 2 + 1](syntactic_input[:, :, self.context_dim:],
                                                                context_input[:, :, :self.context_dim],
                                                                context_input[:, :, :self.context_dim], src_mask),
                            syntactic_input[:, :, self.context_dim:]], dim=2
                        )
                    else:
                        context_output = torch.cat(
                            [context_input[:, :, :self.context_dim],
                             self.attns[atten_layer_idx * 2](context_input[:, :, :self.context_dim],
                                                             syntactic_input[:, :, self.context_dim:],
                                                             syntactic_input[:, :, self.context_dim:], src_mask)], dim=2
                        )
                        syntactic_output = torch.cat([
                            self.attns[atten_layer_idx * 2 + 1](syntactic_input[:, :, self.context_dim:],
                                                                context_input[:, :, :self.context_dim],
                                                                context_input[:, :, :self.context_dim], src_mask),
                            syntactic_input[:, :, self.context_dim:]], dim=2
                        )

            else:
                for i in range(self.opt.get('num_inter_times', 1)):
                    if self.opt['attention'] == 'dep_aware_multi_head':
                        context_output = torch.cat(
                            [context_input[:, :, :self.context_dim],
                             self.attns[atten_layer_idx * 2](context_input[:, :, :self.context_dim],
                                                             syntactic_input[:, :, :self.context_dim],
                                                             syntactic_input[:, :, self.context_dim:],
                                                             dep_rel_smooth, src_mask)], dim=2
                        )
                        syntactic_output = torch.cat([
                            self.attns[atten_layer_idx * 2 + 1](syntactic_input[:, :, self.context_dim:],
                                                                context_input[:, :, self.context_dim:],
                                                                context_input[:, :, :self.context_dim],
                                                                dep_rel_smooth, src_mask),
                            syntactic_input[:, :, self.context_dim:]], dim=2
                        )
                    elif self.opt['attention'] == 'parameter_gen_dep_aware_multi_head':
                        context_output = torch.cat(
                            [context_input[:, :, :self.context_dim],
                             self.attns[atten_layer_idx * 2](context_input[:, :, :self.context_dim],
                                                             syntactic_input[:, :, :self.context_dim],
                                                             syntactic_input[:, :, self.context_dim:],
                                                             src_mask, position_emb)], dim=2
                        )
                        syntactic_output = torch.cat([
                            self.attns[atten_layer_idx * 2 + 1](syntactic_input[:, :, self.context_dim:],
                                                                context_input[:, :, self.context_dim:],
                                                                context_input[:, :, :self.context_dim],
                                                                dep_rel_smooth, self.s2c_meta_weight, src_mask, adj),
                            syntactic_input[:, :, self.context_dim:]], dim=2
                        )
                    elif self.opt['attention'] == 'relative_multi_head':
                        context_output = torch.cat(
                            [context_input[:, :, :self.context_dim],
                             syntactic_input[:, :, self.context_dim:]], dim=2
                        )
                        syntactic_output = torch.cat([
                            self.attns[atten_layer_idx](syntactic_input[:, :, self.context_dim:],
                                                        context_input[:, :, self.context_dim:],
                                                        context_input[:, :, :self.context_dim], src_mask, adj),
                            syntactic_input[:, :, self.context_dim:]], dim=2
                        )
                    elif self.opt['attention'] == 'both_relative_multi_head' or \
                            self.opt['attention'] ==  'both_relative_concat' or \
                            self.opt['attention'] == 'constrain_multi_head':
                        if self.opt.get('ctx_adj', False):
                            ctx_adj = ctx_adj
                        else:
                            ctx_adj = adj
                        if not self.opt.get('ctx_syn_adj_reverse', False):
                            ctx_adj, syn_adj = ctx_adj, adj
                        else:
                            syn_adj, ctx_adj = ctx_adj, adj
                        context_output = torch.cat(
                            [context_input[:, :, :self.context_dim],
                             self.attns[atten_layer_idx * 2](context_input[:, :, :self.context_dim],
                                                             syntactic_input[:, :, :self.context_dim],
                                                             syntactic_input[:, :, self.context_dim:], src_mask, ctx_adj)], dim=2
                        )
                        syntactic_output = torch.cat([
                            self.attns[atten_layer_idx * 2 + 1](syntactic_input[:, :, self.context_dim:],
                                                                context_input[:, :, self.context_dim:],
                                                                context_input[:, :, :self.context_dim], src_mask, syn_adj),
                            syntactic_input[:, :, self.context_dim:]], dim=2
                        )
                    elif self.opt['attention'] == 'syn_local_ctx_relative_multi_head':
                        if self.opt.get('ctx_adj', False):
                            ctx_adj = ctx_adj
                        else:
                            ctx_adj = adj
                        if not self.opt.get('ctx_syn_adj_reverse', False):
                            ctx_adj, syn_adj = ctx_adj, adj
                        else:
                            syn_adj, ctx_adj = ctx_adj, adj
                        context_output = torch.cat(
                            [context_input[:, :, :self.context_dim],
                             self.attns[atten_layer_idx * 2](context_input[:, :, :self.context_dim],
                                                             syntactic_input[:, :, :self.context_dim],
                                                             syntactic_input[:, :, self.context_dim:], src_mask, ctx_adj)], dim=2
                        )
                        syntactic_output = torch.cat([
                            self.attns[atten_layer_idx * 2 + 1](syntactic_input[:, :, self.context_dim:],
                                                                context_input[:, :, self.context_dim:],
                                                                context_input[:, :, :self.context_dim], src_mask),
                            syntactic_input[:, :, self.context_dim:]], dim=2
                        )
                    elif self.opt['attention'] == 'local_relative_multi_head':
                        context_output = torch.cat(
                            [context_input[:, :, :self.context_dim],
                             self.attns[atten_layer_idx * 2](context_input[:, :, :self.context_dim],
                                                             syntactic_input[:, :, :self.context_dim],
                                                             syntactic_input[:, :, self.context_dim:], src_mask, adj)], dim=2
                        )
                        syntactic_output = torch.cat([
                            self.attns[atten_layer_idx * 2 + 1](syntactic_input[:, :, self.context_dim:],
                                                                context_input[:, :, self.context_dim:],
                                                                context_input[:, :, :self.context_dim], src_mask),
                            syntactic_input[:, :, self.context_dim:]], dim=2
                        )
                    else:
                        context_output = torch.cat(
                            [context_input[:, :, :self.context_dim],
                             self.attns[atten_layer_idx * 2](context_input[:, :, :self.context_dim],
                                                             syntactic_input[:, :, :self.context_dim],
                                                             syntactic_input[:, :, self.context_dim:], src_mask)], dim=2
                        )
                        syntactic_output = torch.cat([
                            self.attns[atten_layer_idx * 2 + 1](syntactic_input[:, :, self.context_dim:],
                                                                context_input[:, :, self.context_dim:],
                                                                context_input[:, :, :self.context_dim], src_mask),
                            syntactic_input[:, :, self.context_dim:]], dim=2
                        )
        else:
            context_output, syntactic_output = [torch.cat(
                [context_input[:, :, :self.context_dim],
                 syntactic_input[:, :, self.context_dim:]], dim=2
            ) for _ in range(2)]
        return context_output, syntactic_output

    def context_syn_iteraction(self, context_embs, syntactic_embs, adj, dep_rel_smooth,
                               src_mask, masks, layer_idx, position_emb=None, ctx_adj=None):
        context_input, syntactic_input = context_embs, syntactic_embs
        if not self.opt['weight_share']:
            context_input, syntactic_input = self.dependent_encode(context_input, syntactic_input, adj, masks, layer_idx, ctx_adj=ctx_adj)
            atten_layer_idx = layer_idx - self.opt['num_init_layers']
            context_input, syntactic_input = self.context_syn_att(context_input, syntactic_input, adj, dep_rel_smooth,
                                                                  src_mask, atten_layer_idx, position_emb, ctx_adj=ctx_adj)
        else:
            context_input, syntactic_input = self.dependent_encode(context_input, syntactic_input, adj, masks, 1)
            context_input, syntactic_input = self.context_syn_att(context_input, syntactic_input, adj, dep_rel_smooth,
                                                                  src_mask, 0, position_emb, ctx_adj=ctx_adj)
        # context_input, syntactic_input = self.in_drop(context_input), self.in_drop(syntactic_input)
        context_input, syntactic_input = \
            self.norm_connect(context_input, context_embs, syntactic_input, syntactic_embs, layer_idx)
        return context_input, syntactic_input

    def multi_iteraction(self, context_embs, syntactic_embs, adj, dep_rel_smooth, src_mask, masks, position_emb=None):
        context_list = []
        syntactic_list = []
        infor_list = []

        context_input, syntactic_input = context_embs, syntactic_embs
        if self.opt.get('ctx_adj', False):
            ctx_adj = torch.zeros_like(adj).cuda()
            idx = np.arange(adj.size(1))
            # ctx_adj[:, idx, idx] = 1
            ctx_adj[:, idx[:-1], idx[:-1] + 1] = 1
            ctx_adj[:, idx[:-1] + 1, idx[:-1]] = 1
            ctx_adj = (-1 * (src_mask.type_as(ctx_adj) - 1) * ctx_adj)
        else:
            ctx_adj = None
        for i in range(self.num_layers):
            if i < self.opt['num_init_layers']:
                # continue
                context_input, syntactic_input = \
                    self.context_syn_init_emb(context_input, syntactic_input, adj, masks, i, ctx_adj=ctx_adj)
                infor_list.append(context_input)
            else:
                context_input, syntactic_input = self.context_syn_iteraction(context_input,
                                                                             syntactic_input, adj, dep_rel_smooth,
                                                                             src_mask,
                                                                             masks, i, position_emb, ctx_adj=ctx_adj)
                infor_list.append(torch.cat([context_input, syntactic_input], dim=2))
            context_list.append(context_input)
            syntactic_list.append(syntactic_input)
        if self.opt['dense_connect'] or self.opt.get('elmo_connect', False):
            return context_list, syntactic_list, infor_list
        else:
            return [context_input], [syntactic_input], []

    def forward(self, adj, inputs):
        """
        mask: when the token is not pad, then the value of this token is 0
        src_mask: when the token is not pad, then the value of this token is 1
        """
        words, masks, pos, ner, deprel, head, subj_pos, obj_pos, subj_type, obj_type = inputs  # unpack
        # position_emb = (self.ctx_query_q_position_emb(words), self.ctx_query_k_position_emb(words))
        position_emb = None
        dep_rel_smooth = torch_utils.label_smooth(deprel, self.opt.get('smooth_rate', 0.), self.num_deprel)
        src_mask = (words == constant.PAD_ID).unsqueeze(-2)
        context_embs, syntactic_embs = \
            self.input_emb(words, pos, ner, deprel, subj_pos, obj_pos, masks)

        context_list, syntactic_list, infor_list = self.multi_iteraction(context_embs,
                                                                         syntactic_embs,
                                                                         adj, dep_rel_smooth,
                                                                         src_mask, masks, position_emb)
        if not self.opt.get('ctx_syn_together', False):
            context_aggregate_out = torch.cat(context_list, dim=2)
            syntactic_aggregate_out = torch.cat(syntactic_list, dim=2)
            if self.opt['dense_connect']:
                context_aggregate_out = self.context_aggregate_W(context_aggregate_out)
                syntactic_aggregate_out = self.syntactic_aggregate_W(syntactic_aggregate_out)
            if self.opt.get('elmo_connect', False):
                ctx_elmo_beta = F.softmax((self.ctx_elmo_beta + 1.0) / self.num_layers, dim=0)
                context_aggregate_out = [context_im_out.unsqueeze(-1) for context_im_out in context_list]
                context_aggregate_out = torch.cat(context_aggregate_out, dim=-1)
                context_aggregate_out = self.ctx_elmo_gama * \
                                        torch.matmul(context_aggregate_out, ctx_elmo_beta).squeeze(dim=-1)
                # context_aggregate_out = self.ctx_elmo_transform(context_aggregate_out)

                syn_elmo_beta = F.softmax((self.syn_elmo_beta + 1.0) / self.num_layers, dim=0)
                syntactic_aggregate_out = [syntactic_im_out.unsqueeze(-1) for syntactic_im_out in syntactic_list]
                syntactic_aggregate_out = torch.cat(syntactic_aggregate_out, dim=-1)
                syntactic_aggregate_out = self.syn_elmo_gama * \
                                          torch.matmul(syntactic_aggregate_out, syn_elmo_beta).squeeze(dim=-1)
            # syntactic_aggregate_out = self.syn_elmo_transform(syntactic_aggregate_out)
        if self.opt.get('merge', 'concat') == 'ali_merge':
            mi_out = torch.cat([context_aggregate_out, syntactic_aggregate_out,
                                (context_aggregate_out - syntactic_aggregate_out)], dim=2)
            mi_out = F.sigmoid(self.ali_gate(mi_out)) * mi_out
        elif self.opt.get('merge', 'concat') == 'highway':
            mi_out = torch.cat([context_aggregate_out, syntactic_aggregate_out], dim=2)
            for i in range(self.opt.get('highway_layers'), 1):
                highway_gate = F.sigmoid(self.highway_gate(mi_out))
                highway_tran = F.relu(self.highway_transform(mi_out))
                mi_out = torch.mul(highway_gate, highway_tran) + torch.mul((1 - highway_gate), mi_out)
        elif self.opt.get('merge', 'concat') == 'gate_merge':
            if not self.opt.get('ctx_syn_together', False):
                mi_out = torch.cat([context_aggregate_out, syntactic_aggregate_out], dim=2)
                gate = self.gate(mi_out)
                mi_out = torch.mul(gate, mi_out)
            else:
                del context_list, syntactic_list
                mi_out = torch.cat(infor_list, dim=2)
                gate = self.gate(mi_out)
                mi_out = torch.mul(gate, mi_out)
        elif self.opt.get('merge', 'concat') == 'entity_gate_merge':
            mi_out = torch.cat([context_aggregate_out, syntactic_aggregate_out], dim=2)
            subj_mask, obj_mask = subj_pos.eq(0).eq(0).unsqueeze(2) + masks.unsqueeze(2), \
                                  obj_pos.eq(0).eq(0).unsqueeze(2) + masks.unsqueeze(2)  # invert mask
            subj_mi = pool(mi_out, subj_mask, type='max')
            obj_mi = pool(mi_out, obj_mask, type='max')
            gate = self.gate(torch.cat([subj_mi, obj_mi], dim=1)).unsqueeze(1)
            mi_out = torch.mul(gate, mi_out)
        else:
            if not self.opt.get('ctx_syn_together', False):
                mi_out = torch.cat([context_aggregate_out, syntactic_aggregate_out], dim=2)
            else:
                mi_out = torch.cat(infor_list, dim=2)
        mi_out = self.ctx_syn_aggregate_W(mi_out)
        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)
        return mi_out, mask


class OriginGraphConvLayer(nn.Module):
    """
    A GCN module operated on dependency graphs.
    syntactic representation
    """

    def __init__(self, opt, mem_dim, out_dim, layers):
        super(OriginGraphConvLayer, self).__init__()
        self.opt = opt
        self.mem_dim = mem_dim
        self.out_dim = out_dim
        self.layers = layers
        self.gcn_drop = nn.Dropout(opt['gcn_dropout'])
        self.linear_output = nn.Linear(self.mem_dim, self.out_dim)
        self.weight_list = nn.ModuleList()
        for i in range(self.layers):
            self.weight_list.append(nn.Linear(self.mem_dim, self.mem_dim))

        if self.opt['cuda']:
            self.weight_list = self.weight_list.cuda()
            self.linear_output = self.linear_output.cuda()

    def forward(self, adj, gcn_inputs):
        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1
        outputs = gcn_inputs
        for l in range(self.layers):
            Ax = adj.bmm(outputs)
            AxW = self.weight_list[l](Ax)
            AxW = AxW + self.weight_list[l](outputs)  # self loop
            AxW = AxW / denom
            gAxW = F.relu(AxW)
            outputs = self.gcn_drop(gAxW)
        outputs = self.linear_output(outputs + gcn_inputs)
        return outputs


class GraphConvLayer(nn.Module):
    """
    A GCN module operated on dependency graphs.
    syntactic representation
    """

    def __init__(self, opt, mem_dim, out_dim, layers):
        super(GraphConvLayer, self).__init__()
        self.opt = opt
        self.mem_dim = mem_dim
        self.out_dim = out_dim
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.gcn_drop = nn.Dropout(opt['gcn_dropout'])

        # linear transformation
        self.linear_output = nn.Linear(self.mem_dim, self.out_dim)

        # dcgcn block
        self.weight_list = nn.ModuleList()
        for i in range(self.layers):
            self.weight_list.append(nn.Linear((self.mem_dim + self.head_dim * i), self.head_dim))

        if self.opt['cuda']:
            self.weight_list = self.weight_list.cuda()
            self.linear_output = self.linear_output.cuda()

    def forward(self, adj, gcn_inputs):
        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1

        outputs = gcn_inputs
        cache_list = [outputs]
        output_list = []

        for l in range(self.layers):
            Ax = adj.bmm(outputs)
            AxW = self.weight_list[l](Ax)
            AxW = AxW + self.weight_list[l](outputs)  # self loop
            AxW = AxW / denom
            gAxW = F.relu(AxW)
            # gAxW = F.leaky_relu(AxW)
            cache_list.append(gAxW)
            outputs = torch.cat(cache_list, dim=2)
            output_list.append(self.gcn_drop(gAxW))

        gcn_outputs = torch.cat(output_list, dim=2)
        gcn_outputs = gcn_outputs + gcn_inputs
        out = self.linear_output(gcn_outputs)

        return out


def pool(h, mask, type='max'):
    if type == 'max':
        h = h.masked_fill(mask, -constant.INFINITY_NUMBER)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])



