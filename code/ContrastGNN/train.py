import math
import os
from collections import defaultdict
from itertools import product
import random

import dill
from dotmap import DotMap

import numpy as np
import pandas as pd
# pd.set_option('display.max_rows', 100)
# pd.set_option('display.min_rows', 100)
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn import metrics
from sklearn.neighbors import kneighbors_graph

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_default_dtype(torch.float32)
torch.multiprocessing.set_sharing_strategy('file_system')

import dgl
import dgl.nn
import dgl.function as fn
from dgl.nn import RelGraphConv
from dgl.utils import expand_as_pair
from dgl.nn import edge_softmax

from my_nn import GraphConv, GATConv, HeteroGraphConv
from load_label import *
from model import *
from predictor import *
from augmentation import *

seed = 100
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# BitCoinAlpha / BitCoinOTC
dataset_name = 'BitCoinAlpha'

############################################### Load user and features ############################################################
het_num_nodes_dict = {}
het_node_feat_dict = {}
het_data_dict = {}
het_edge_feat_dict = {}

with open(f"./data/{dataset_name}/user_dict.pkl", "rb") as dill_file:
    dict_user2id = dill.load(dill_file)
het_num_nodes_dict['user'] = len(dict_user2id)

if dataset_name == 'BitCoinAlpha' or dataset_name == 'BitCoinOTC':
    user_feat = np.random.rand(len(dict_user2id), 64)
    het_node_feat_dict['user'] = user_feat
    
print(het_node_feat_dict['user'].shape)

############################################### Load Train Graph ############################################################
with open(f"./data/{dataset_name}/g_train.pkl", "rb") as dill_file:
    tmp_het_data_dict = dill.load(dill_file)
    tmp_het_edata_dict = dill.load(dill_file)
het_data_dict.update(tmp_het_data_dict)

for k in tmp_het_edata_dict:
    het_edge_feat_dict[k] = torch.from_numpy(tmp_het_edata_dict[k])

graph_user = dgl.heterograph(
    data_dict = het_data_dict,
    num_nodes_dict = het_num_nodes_dict
)
for node_t in het_node_feat_dict:
    graph_user.nodes[node_t].data['feature'] = torch.from_numpy(het_node_feat_dict[node_t]).float()


############################################### Training Parameter Setting ############################################################

args = DotMap()
args.num_nodes = graph_user.num_nodes()
# ['FriendApplyAgree', 'TeamInviteAgree', 'GameWatchGift', 'ZoneLeaveMessageGift', 'PrivateChat', 'GameSendLike', 'FriendApplyRefuse', 'Report']
args.pos_edge_type = ['positive']
args.neg_edge_type = ['negative']
args.edge_type = args.pos_edge_type+args.neg_edge_type
args.num_edge_types = len(args.edge_type)
args.dim_features = graph_user.nodes['user'].data['feature'].shape[1]
args.dim_hiddens = 128
args.dim_embs = 128

args.learning_rate = 0.01

args.conv_depth = 2
args.loss_batch_size = 102400             # to calculated loss

#args.inference_batch_size = 128       # the batch size for inferencing all/batched nodes embeddings
args.sampling_batch_size = 128
args.residual = False
args.num_heads = 8
args.dropout = 0

# active_tag walktogether_tag  friend_tag  playagain_tag 
# label
args.label = 'label'  
args.conv_type = 'gat'
args.het_agg_type = 'attn' # multiplex aggregation
args.dim_query = 128
args.predictor = '2-linear'
# concat / mean / attn / pos
args.combine_type = 'concat'

# sign / common
args.sign_conv = 'sign'
# pos / neg / both
args.sign_aggre = 'both'
# pos / neg / intra / inter / all
args.contrast_type = 'all'
# delete / change / reverse / composite
args.augment = 'change'

#args.contrastive = True
args.mask_ratio = 0.1
args.tao = 0.05
args.alpha = 1e-4
args.beta = 0.8
args.pos_gamma = 1
args.neg_gamma = 1

args.gpu = 0
args.num_workers = 0
args.verbose = 1
args.pretrain_epochs = 600
args.finetune_epochs = 0
# both / pos / neg
args.drop_type = 'both'

# 2-layer 20

device = torch.device(f'cuda:{args.gpu}')

with open(f"./data/{dataset_name}/{args.label}.pkl", "rb") as dill_file:
    label_train = dill.load(dill_file)
    label_test = dill.load(dill_file)
    
label_ids = np.unique(np.concatenate((label_train.src, label_train.dst, label_test.src, label_test.dst)))


############################################### Training ############################################################

test_results = []

for m in range(1):
    g = graph_user.edge_type_subgraph(args.edge_type)
    pos_weight = None
    model = Model(args).to(device)
    opt = torch.optim.Adam(model.parameters())

    dataset = NodeBatch(np.arange(g.num_nodes()))
    dataloader_nodes = torch.utils.data.DataLoader(dataset, batch_size=args.loss_batch_size, shuffle=True)


    dataset = LabelPairs(label_train)
    dataloader_labels = torch.utils.data.DataLoader(dataset, batch_size=int(args.loss_batch_size*len(label_train)/g.num_nodes()), shuffle=True)
    res = defaultdict(list)

    for e in range(args.pretrain_epochs+args.finetune_epochs):
        g_attr, g_stru = GraphAug(g, args)
        cnt = 0

        for nids, (pair, y) in zip(dataloader_nodes, dataloader_labels):
            u, v = pair.T
            y = y.to(device)
            nids = torch.unique(torch.cat((u, v), dim=-1))

            
            embs_attr_pos, embs_stru_pos, embs_attr_neg, embs_stru_neg = model(g_attr, g_stru, nids, device)
            loss_contrastive = model.compute_contrastive_loss(device, embs_attr_pos[nids], embs_stru_pos[nids], embs_attr_neg[nids], embs_stru_neg[nids])

            y_score = model.predict_combine((embs_attr_pos,embs_stru_pos,embs_attr_neg, embs_stru_neg), u, v, device)
            loss_label = model.compute_label_loss(y_score, y, pos_weight, device)

            loss = args.alpha  * loss_contrastive + loss_label

            #print(f'epoch:{e}  {cnt}/{len(dataloader_nodes)}  loss_contrastive:{loss_contrastive}  loss_label:{loss_label}.')

            opt.zero_grad()
            loss.backward()
            opt.step()

            cnt += 1
            
            if (e+1) % 50 ==0:
                print(f'epoch:{e}  {cnt}/{len(dataloader_nodes)}  loss_contrastive:{loss_contrastive}  loss_label:{loss_label}.')
                with torch.no_grad():
                    embs, (attn_attr_pos, attn_stru_pos, attn_attr_neg, attn_stru_neg) = model.inference(g, g, label_ids, device)
                    train_auc, train_prec, train_recl, train_micro_f1, train_binary_f1, train_macro_f1 = eval_metric(embs, model, label_train, args, device)
                    test_auc, test_prec, test_recl, test_micro_f1, test_binary_f1, test_macro_f1 = eval_metric(embs, model, label_test, args, device)

                    res['train'].append([train_auc, train_prec, train_recl, train_micro_f1, train_binary_f1, train_macro_f1])
                    res['test'].append([test_auc, test_prec, test_recl, test_micro_f1, test_binary_f1, test_macro_f1])
                    print(f'Epoch {e}, {cnt}/{len(dataloader_nodes)}.')
                    print(f'Training (AUC, Precision, Recall, Micro_F1, Binary_F1, Macro_F1):      ({train_auc:.4f}, {train_prec:.4f}, {train_recl:.4f}, {train_micro_f1:.4f}, {train_binary_f1:.4f}, {train_macro_f1:.4f})')
                    print(f'Testing  (AUC, Precision, Recall, Micro_F1, Binary_F1, Macro_F1):      ({test_auc:.4f}, {test_prec:.4f}, {test_recl:.4f}, {test_micro_f1:.4f}, {test_binary_f1:.4f}, {test_macro_f1:.4f})')
        torch.cuda.empty_cache()
    print(f'repeat: {m}')
    test_results.append([test_auc, test_micro_f1, test_binary_f1, test_macro_f1])
    
print(f'mean auc, micro_f1, binary_f1, macro_f1:{np.array(test_results).sum(0)/np.array(test_results).shape[0]}')
