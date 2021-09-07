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

############################################### Load Labels ############################################################
class LabelPairs(torch.utils.data.Dataset):
    def __init__(self, df):
        super(LabelPairs).__init__()
        u = torch.from_numpy(df.src.values).long()
        v = torch.from_numpy(df.dst.values).long()
        y = torch.from_numpy(df['label'].values).double()
        self.pairs = torch.stack((u, v), dim=0)
        self.label = y
    
    def __getitem__(self, index):
        return self.pairs[:, index], self.label[index]
    
    def __len__(self):
        return len(self.label)

class NodeBatch(torch.utils.data.Dataset):
    def __init__(self, nodes):
        self.nodes = torch.from_numpy(nodes)
    
    def __getitem__(self, index):
        return self.nodes[index]
    
    def __len__(self):
        return len(self.nodes)

############################################### Define Model ############################################################
"""Torch modules for graph convolutions(GCN)."""
# pylint: disable= no-member, arguments-differ, invalid-name

from torch import nn
from torch.nn import init
from torch.functional import F

from dgl import function as fn
from dgl.base import DGLError
from dgl.utils import expand_as_pair
from dgl.nn import edge_softmax
from dgl.nn import utils


'''
Define Hetero Layer
'''
class HetAttn(nn.Module):
    def __init__(self, args, etypes):
        super(HetAttn, self).__init__()
        self.args = DotMap(args.toDict())
        args = self.args
        self.etypes = etypes
        self.feature_trans = nn.Linear(args.dim_features, args.dim_hiddens, bias=False)
        self.convs = nn.ModuleList()
        
        if self.args.conv_type == 'gcn':
            for _ in range(args.conv_depth - 1):
                self.convs.append(HeteroGraphConv({rel: GraphConv(args.dim_hiddens, args.dim_hiddens, allow_zero_in_degree=True, residual=args.residual)
                                                   for rel in self.etypes},
                                  agg_type=args.het_agg_type, dim_values=args.dim_hiddens, dim_query=args.dim_query)) 
            self.convs.append(HeteroGraphConv({rel: GraphConv(args.dim_hiddens, args.dim_embs, allow_zero_in_degree=True, residual=args.residual) 
                                               for rel in self.etypes},
                              agg_type=args.het_agg_type, dim_values=args.dim_embs, dim_query=args.dim_query))
        elif self.args.conv_type == 'gat':
            for _ in range(args.conv_depth - 1):
                self.convs.append(HeteroGraphConv({rel: GATConv(args.dim_hiddens, args.dim_hiddens // args.num_heads, args.num_heads, allow_zero_in_degree=True, residual=args.residual)
                                                   for rel in self.etypes},
                                  agg_type=args.het_agg_type, dim_values=args.dim_hiddens, dim_query=args.dim_query)) 
            self.convs.append(HeteroGraphConv({rel: GATConv(args.dim_hiddens, args.dim_embs // args.num_heads, args.num_heads, allow_zero_in_degree=True, residual=args.residual) 
                                               for rel in self.etypes},
                          agg_type=args.het_agg_type, dim_values=args.dim_embs, dim_query=args.dim_query))
        
        self.concat_weight = nn.Linear((args.conv_depth + 1) * args.dim_embs, args.dim_embs, bias=False)

        if self.args.conv_depth == 1:
            self.sampler_nodes = [5]
            self.sampler_inference = [10]
        elif self.args.conv_depth == 2:
            self.sampler_nodes = [10, 20]
            self.sampler_inference = [10 , 20]
        else:
            raise
#         if self.args.conv_depth == 3:
#             self.sampler_nodes = [5, 10, 10]
#             self.sampler_inference = [10, 10, 10]
            
        nn.init.xavier_uniform_(self.feature_trans.weight)
        
        
    def calc_from_loader(self, loader, x, device):
        y = torch.zeros(len(x), self.args.dim_embs)
        attn_res = torch.zeros(len(x), len(self.etypes))
        
        def calc_from_blocks(blocks, conv_idx, x, device):
            input_nodes, output_nodes = blocks[0].srcdata[dgl.NID], blocks[0].dstdata[dgl.NID]
            h = x[input_nodes].to(device)
            h = torch.tanh(self.feature_trans(h))
            for b, idx in zip(blocks, conv_idx):
                b = b.to(device)
                h, attn = self.convs[idx](b, h)
            return h, attn
        
        for input_nodes, output_nodes, blocks in loader:
            
            h0 = x[output_nodes].to(device)
            h0 = torch.tanh(self.feature_trans(h0))
            emb_ulti = [h0]
            if self.args.conv_depth == 1:
                h1, attn = calc_from_blocks(blocks, [0], x, device)
                emb_ulti.append(h1)
            if self.args.conv_depth ==2:
                h1, _ = calc_from_blocks(blocks[1:], [1], x, device)
                h2, attn = calc_from_blocks(blocks, [0, 1], x, device)
                emb_ulti.extend([h1, h2])
            
            y[output_nodes] = self.concat_weight(torch.cat(emb_ulti, dim=-1)).cpu()
            attn_res[output_nodes] = attn.squeeze(dim=-1).transpose(0, 1).cpu()
        return y, attn_res
        

    def forward(self, g, x, nids, device):
        dataloader = dgl.dataloading.NodeDataLoader(g, nids,
                                                    dgl.dataloading.MultiLayerNeighborSampler(self.sampler_nodes),
                                                    batch_size=self.args.sampling_batch_size,
                                                    num_workers=self.args.num_workers,
                                                    shuffle=True,
                                                    drop_last=False)
        y, attn_res = self.calc_from_loader(dataloader, x, device)
        return y, attn_res
    
    
    def inference(self, g, x, nids, device):
#         dataloader = dgl.dataloading.NodeDataLoader(g, nids,
#                                                     dgl.dataloading.MultiLayerFullNeighborSampler(len(self.sampler_nodes)),
#                                                     batch_size=self.args.inference_batch_size,
#                                                     num_workers=self.args.num_workers,
#                                                     shuffle=True,
#                                                     drop_last=False)
        dataloader = dgl.dataloading.NodeDataLoader(g, nids,
                                                    dgl.dataloading.MultiLayerNeighborSampler(self.sampler_inference),
                                                    batch_size=self.args.sampling_batch_size,
                                                    num_workers=self.args.num_workers,
                                                    shuffle=True,
                                                    drop_last=False)
        y, attn_res = self.calc_from_loader(dataloader, x, device)
        return y, attn_res
    

'''
Define Whole Model
'''

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.sign_conv == 'sign':
            self.pos_emb_model = HetAttn(args, args.pos_edge_type)
            self.neg_emb_model = HetAttn(args, args.neg_edge_type)
        elif args.sign_conv == 'common':
            self.emb_model = HetAttn(args, args.edge_type)
        self.args = args
        self.link_predictor = ScorePredictor(args, dim_embs = args.dim_embs)
        
        self.combine_type = args.combine_type
        
        if self.args.sign_aggre!='both':
            transform_type = 2
        elif self.args.sign_aggre == 'both' or self.args.sign_conv == 'common':
            transform_type = 4
        
        if self.combine_type == 'concat':
            self.transform = nn.Sequential(nn.Linear(transform_type*args.dim_embs, args.dim_embs))
            self.link_predictor = ScorePredictor(args, dim_embs = args.dim_embs)
        elif self.combine_type == 'attn':
            self.attention = nn.Sequential(nn.Linear(args.dim_embs, args.dim_query), nn.Tanh(), nn.Linear(agrs.dim_query, 1, bias=False))
            self.link_predictor = ScorePredictor(args, dim_embs=args.dim_embs)
        
    def forward(self, g_attr, g_stru, nids, device):
#         nids = torch.unique(torch.cat((uids, vids), dim=-1))
#         embs_pos, embs_neg = self.emb_model(g, x, nids, device)
#         score = self.predict_combine((embs_pos, embs_neg), uids, vids, device)
        if self.args.sign_conv == 'common':
            embs_attr_pos, _ = self.emb_model(g_attr.edge_type_subgraph(self.args.pos_edge_type), g_attr.ndata['feature'], nids, device)
            embs_stru_pos, _ = self.emb_model(g_stru.edge_type_subgraph(self.args.pos_edge_type), g_stru.ndata['feature'], nids, device)
            embs_attr_neg, _ = self.emb_model(g_attr.edge_type_subgraph(self.args.neg_edge_type), g_attr.ndata['feature'], nids, device)
            embs_stru_neg, _ = self.emb_model(g_stru.edge_type_subgraph(self.args.neg_edge_type), g_stru.ndata['feature'], nids, device)
            return embs_attr_pos, embs_stru_pos, embs_attr_neg, embs_stru_neg
        elif self.args.sign_conv == 'sign':
            embs_attr_pos, _ = self.pos_emb_model(g_attr.edge_type_subgraph(self.args.pos_edge_type), g_attr.ndata['feature'], nids, device)
            embs_attr_neg, _ = self.neg_emb_model(g_attr.edge_type_subgraph(self.args.neg_edge_type), g_attr.ndata['feature'], nids, device)
            embs_stru_pos, _ = self.pos_emb_model(g_stru.edge_type_subgraph(self.args.pos_edge_type), g_stru.ndata['feature'], nids, device)
            embs_stru_neg, _ = self.neg_emb_model(g_stru.edge_type_subgraph(self.args.neg_edge_type), g_stru.ndata['feature'], nids, device)
            return embs_attr_pos, embs_stru_pos, embs_attr_neg, embs_stru_neg
    
    def inference(self, g_attr, g_stru, nids, device):
        if self.args.sign_conv == 'common':
            embs_attr_pos, attn_attr_pos = self.emb_model(g_attr, g_attr.ndata['feature'], nids, device)
            embs_attr_neg, attn_attr_neg = self.emb_model(g_attr, g_attr.ndata['feature'], nids, device)
            embs_stru_pos, attn_stru_pos = self.emb_model(g_stru, g_stru.ndata['feature'], nids, device)
            embs_stru_neg, attn_stru_neg = self.emb_model(g_stru, g_stru.ndata['feature'], nids, device)
            return (embs_attr_pos, embs_stru_pos, embs_attr_neg, embs_stru_neg), (attn_attr_pos, attn_stru_pos, attn_attr_neg, attn_stru_neg)
        elif self.args.sign_conv == 'sign':
            embs_attr_pos, attn_attr_pos = self.pos_emb_model(g_attr.edge_type_subgraph(self.args.pos_edge_type), g_attr.ndata['feature'], nids, device)
            embs_attr_neg, attn_attr_neg = self.neg_emb_model(g_attr.edge_type_subgraph(self.args.neg_edge_type), g_attr.ndata['feature'], nids, device)
            embs_stru_pos, attn_stru_pos = self.pos_emb_model(g_stru.edge_type_subgraph(self.args.pos_edge_type), g_stru.ndata['feature'], nids, device)
            embs_stru_neg, attn_stru_neg = self.neg_emb_model(g_stru.edge_type_subgraph(self.args.neg_edge_type), g_stru.ndata['feature'], nids, device)
            return (embs_attr_pos, embs_stru_pos, embs_attr_neg, embs_stru_neg), (attn_attr_pos, attn_stru_pos, attn_attr_neg, attn_stru_neg)

        
    def compute_contrastive_loss(self, device, embs_attr_pos, embs_stru_pos, embs_attr_neg=None, embs_stru_neg=None):
        nodes_num = embs_attr_pos.shape[0]
        feature_size = embs_attr_pos.shape[1]
        
        embs_attr_pos = embs_attr_pos.to(device)
        embs_stru_pos = embs_stru_pos.to(device)
        normalized_embs_attr_pos = F.normalize(embs_attr_pos, p=2, dim=1)
        normalized_embs_stru_pos = F.normalize(embs_stru_pos, p=2, dim=1)
        if embs_attr_neg!=None and embs_stru_neg!=None:
            embs_attr_neg = embs_attr_neg.to(device)
            embs_stru_neg = embs_stru_neg.to(device)
            normalized_embs_attr_neg = F.normalize(embs_attr_neg, p=2, dim=1)
            normalized_embs_stru_neg = F.normalize(embs_stru_neg, p=2, dim=1)
        
        
        def inter_contrastive(embs_attr, embs_stru):
            pos = torch.exp(torch.div(torch.bmm(embs_attr.view(nodes_num, 1, feature_size), embs_stru.view(nodes_num, feature_size, 1)), self.args.tao))
            
            def generate_neg_score(embs_1, embs_2):
                neg_similarity = torch.mm(embs_1.view(nodes_num, feature_size), embs_2.transpose(0,1))
                neg_similarity[np.arange(nodes_num),np.arange(nodes_num)] = 0
                return torch.sum(torch.exp(torch.div( neg_similarity  , self.args.tao)) , dim=1)
            
            neg = generate_neg_score(embs_attr, embs_stru)

            return torch.mean(- (torch.log(torch.div(pos, neg))))
        
        def intra_contrastive(self_embs, embs_attr_pos, embs_attr_neg, embs_stru_pos, embs_stru_neg):
            pos_score_1 = torch.exp(torch.div(torch.bmm(self_embs.view(nodes_num, 1, feature_size), embs_attr_pos.view(nodes_num, feature_size, 1)), self.args.tao))
            pos_score_2 = torch.exp(torch.div(torch.bmm(self_embs.view(nodes_num, 1, feature_size), embs_stru_pos.view(nodes_num, feature_size, 1)), self.args.tao))
            pos = pos_score_1 + pos_score_2
            def generate_neg_score(pos_embs, neg_embs_1, neg_embs_2):
                neg_score_1 = torch.bmm(pos_embs.view(nodes_num, 1, feature_size), neg_embs_1.view(nodes_num, feature_size, 1))
                neg_score_2 = torch.bmm(pos_embs.view(nodes_num, 1, feature_size), neg_embs_2.view(nodes_num, feature_size, 1))
                return torch.exp(torch.div(neg_score_1, self.args.tao)) + torch.exp(torch.div(neg_score_2, self.args.tao))
            neg = generate_neg_score(self_embs, embs_attr_neg, embs_stru_neg)
            return torch.mean(- torch.log(torch.div(pos, neg)) )
            

        inter_pos = inter_contrastive(normalized_embs_attr_pos, normalized_embs_stru_pos)
        inter_neg = inter_contrastive(normalized_embs_attr_neg, normalized_embs_stru_neg)
        
        embs = torch.cat((embs_attr_pos,embs_stru_pos,embs_attr_neg, embs_stru_neg), dim=-1)
        self_embs = self.transform(embs)
        normalized_self_embs = F.normalize(self_embs, p=2, dim=1)
        
        intra = intra_contrastive(normalized_self_embs, normalized_embs_attr_pos, normalized_embs_attr_neg, normalized_embs_stru_pos, normalized_embs_stru_neg)
        #print(f'inter_pos:{inter_pos}  inter_neg:{inter_neg}  intra:{intra}')
        if self.args.contrast_type == 'pos':
            return inter_pos
        elif self.args.contrast_type == 'neg':
            return inter_neg
        elif self.args.contrast_type == 'intra':
            return intra
        elif self.args.contrast_type == 'inter':
            return inter_pos + inter_neg
        elif self.args.contrast_type == 'all':
            return (1-self.args.beta) * (inter_pos + inter_neg) + self.args.beta * intra
            

        
    
    def compute_label_loss(self, score, y_label, pos_weight, device):
        pos_weight = torch.tensor([(y_label==0).sum().item()/(y_label==1).sum().item()]*y_label.shape[0]).to(device)
        return F.binary_cross_entropy_with_logits(score, y_label, pos_weight=pos_weight)
    
        
        
    def predict_combine(self, embs, uids, vids, device):
        u_embs = self.combine(embs, uids, device)
        v_embs = self.combine(embs, vids, device)
        score = self.link_predictor(u_embs, v_embs)
        return score
    
    def compute_attention(self, embs):
        attn = self.attention(embs).softmax(dim=0)
        return attn
    
    def combine(self, embs, nids, device):
        if self.args.sign_conv == 'sign':
            if self.args.sign_aggre == 'pos':
                embs = (embs[0],embs[1])
            elif self.args.sign_aggre == 'neg':
                embs = (embs[2],embs[3])
            
        if self.combine_type == 'concat':
            embs = torch.cat(embs, dim=-1)
            sub_embs = embs[nids].to(device)
            out_embs = self.transform(sub_embs)
            return out_embs
        elif self.combine_type == 'attn':
            embs = torch.stack(embs, dim=0)
            sub_embs = embs[:,nids].to(device)
            attn = self.compute_attention(sub_embs)
            # attn: (2,n,1)   sub_embs: (2,n,feature)
            out_embs = (attn*sub_embs).sum(dim=0)
            return our_embs
        elif self.combine_type == 'mean':
            embs = torch.stack(embs, dim=0).mean(dim=0)
            sub_embs = embs[nids].to(device)
            return sub_embs
        elif self.combine_type == 'pos':
            sub_embs = embs[0][nids].to(device)
            return sub_embs

############################################### Define Predictor ############################################################
class ScorePredictor(nn.Module):
    def __init__(self, args, **params):
        super().__init__()
        self.args = DotMap(args.toDict())
        for k,v in params.items():
            self.args[k] = v
        
        if self.args.predictor == 'dot':
            pass
        elif self.args.predictor == '1-linear':
            self.predictor = nn.Linear(self.args.dim_embs*2, 1)
        elif self.args.predictor == '2-linear':
            self.predictor = nn.Sequential(nn.Linear(self.args.dim_embs*2, self.args.dim_embs),
                                          nn.LeakyReLU(),
                                          nn.Linear(self.args.dim_embs, 1))
        elif self.args.predictor == '3-linear':
            self.predictor = nn.Sequential(nn.Linear(self.args.dim_embs*2, self.args.dim_embs),
                                          nn.LeakyReLU(),
                                          nn.Linear(self.args.dim_embs, self.args.dim_embs),
                                          nn.LeakyReLU(),
                                          nn.Linear(self.args.dim_embs, 1)
                                         )
        elif self.args.predictor == '4-linear':
            self.predictor = nn.Sequential(nn.Linear(self.args.dim_embs*2, self.args.dim_embs),
                                          nn.LeakyReLU(),
                                          nn.Linear(self.args.dim_embs, self.args.dim_embs),
                                          nn.LeakyReLU(),
                                          nn.Linear(self.args.dim_embs, self.args.dim_embs),
                                          nn.LeakyReLU(),
                                          nn.Linear(self.args.dim_embs, 1)
                                         )
        self.reset_parameters()
            
    def reset_parameters(self):
        pass

    def forward(self, u_e, u_v):
        if self.args.predictor == 'dot':
            score = u_e.mul(u_v).sum(dim=-1)
        else:
            x = torch.cat([u_e, u_v], dim=-1)
            score = self.predictor(x).flatten()
        return score

def eval_model(embs, model, df, batched, args, device):
    if batched:
        dataset = LabelPairs(df)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.loss_batch_size, num_workers=args.num_workers, shuffle=True)
        y_pre_list = []
        y_true_list = []
        for pair, y in dataloader:
            uids, vids = pair.T
            score = model.predict_combine(embs, uids, vids, device)
            y_pre_list.append(torch.sigmoid(score))
            y_true_list.append(y)
        y_pre = torch.cat(y_pre_list, dim=-1).cpu().numpy()
        y_true = torch.cat(y_true_list, dim=-1).cpu().numpy()
    else:
        uids = torch.from_numpy(df.src.values).long()
        vids = torch.from_numpy(df.dst.values).long()
        score = model.predict_combine(embs, uids, vids, device)
        y_pre = torch.sigmoid(score).cpu().numpy()
        y_true = df['label'].values
    return y_true, y_pre
    
def eval_metric(embs, model, df, args, device, threshold=0.05):
	# change threshold according to different datasets
	# 0.05 for Alpha, 0.1 for OTC
    y_true, y_pre = eval_model(embs, model, df, args.eval_batched, args, device)
    y = (y_pre > threshold)
    auc = metrics.roc_auc_score(y_true, y_pre)
    prec = metrics.precision_score(y_true, y)
    recl = metrics.recall_score(y_true, y)
    binary_f1 = metrics.f1_score(y_true, y, average='binary')
    micro_f1 = metrics.f1_score(y_true, y, average='micro')
    macro_f1 = metrics.f1_score(y_true, y, average='macro')
    
    
    return auc, prec, recl, micro_f1, binary_f1, macro_f1

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


############################################### Graph Augmentation ############################################################

def generate_mask(mask_ratio, row, column):
    # 1 -- leave   0 -- drop
    arr_mask_ratio = np.random.uniform(0,1,size=(row, column))
    arr_mask = np.ma.masked_array(arr_mask_ratio, mask=(arr_mask_ratio<mask_ratio)).filled(0)
    arr_mask = np.ma.masked_array(arr_mask, mask=(arr_mask>=mask_ratio)).filled(1)
    return arr_mask

def generate_attr_graph(g, args):
    # generate noise g_attr
    feature = g.ndata['feature']
    attr_noise = np.random.normal(loc=0, scale=0.1, size=(feature.shape[0], feature.shape[1]))
    attr_mask = generate_mask(args.mask_ratio, row=feature.shape[0], column=feature.shape[1])
    noise_feature = feature*attr_mask + (1-attr_mask) * attr_noise
    
    g_attr = g
    g_attr.ndata['feature'] = noise_feature.float()
    return g_attr

def generate_stru_graph(g, args):
    # generate noise g_stru by deleting links
    g_stru = g

    if args.drop_type == 'both':
        edge_types = args.edge_type
    elif args.drop_type == 'pos':
        edge_types = args.pos_edge_type
    elif args.drop_type == 'neg':
        edge_types = args.neg_edge_type
        
    for etype in edge_types:
        etype_edges = g.edges(etype=etype)
        # shape: (e, 2)
        df = np.array([etype_edges[0].numpy(), etype_edges[1].numpy()]).transpose()
        
        # delete edges
        edge_mask = generate_mask(args.mask_ratio, row=1, column=len(etype_edges[0]))
        drop_eids = torch.arange(0,len(etype_edges[0]))[edge_mask==0]
        g_stru = dgl.remove_edges(g_stru, drop_eids, etype=etype)

        # add an equal number of edges
        add_row = []
        add_column = []
        index = 0
        while index < len(drop_eids):
            row_sample = np.random.randint(g.num_nodes())
            column_sample = np.random.randint(g.num_nodes())
            if (df==[row_sample, column_sample]).all(1).any() == False:
                index += 1
                add_row.append(row_sample)
                add_column.append(column_sample)
        g_stru = dgl.add_edges(g_stru, add_row, add_column, etype=etype)

    g_stru.ndata['feature'] = g_stru.ndata['feature'].float()
    return g_stru


def generate_stru_sign_graph(g, args):
    # generate noise g_stru by exchanging some pos/neg links
    g_stru = g
    
    if args.drop_type == 'both':
        edge_types = args.edge_type
    elif args.drop_type == 'pos':
        edge_types = args.pos_edge_type
    elif args.drop_type == 'neg':
        edge_types = args.neg_edge_type
    
    for etype in edge_types:
        etype_edges = g.edges(etype=etype)
        edge_mask = generate_mask(args.mask_ratio, row=1, column=len(etype_edges[0]))
        
        # delete edges
        drop_eids = torch.arange(0,len(etype_edges[0]))[edge_mask==0]
        g_stru = dgl.remove_edges(g_stru, drop_eids, etype=etype)
        
        # add_edges
        if etype in args.pos_edge_type:
            g_stru = dgl.add_edges(g_stru, etype_edges[0][drop_eids], etype_edges[1][drop_eids] , etype=random.choice(args.neg_edge_type))
        elif etype in args.neg_edge_type:
            g_stru = dgl.add_edges(g_stru, etype_edges[0][drop_eids], etype_edges[1][drop_eids] , etype=random.choice(args.pos_edge_type))
    g_stru.ndata['feature'] = g_stru.ndata['feature'].float()
    return g_stru

def generate_stru_status_graph(g, args):
    g_stru = g
    
    if args.drop_type == 'both':
        edge_types = args.edge_type
    elif args.drop_type == 'pos':
        edge_types = args.pos_edge_type
    elif args.drop_type == 'neg':
        edge_types = args.neg_edge_type
    
    for etype in edge_types:
        etype_edges = g.edges(etype=etype)
        edge_mask = generate_mask(args.mask_ratio, row=1, column=len(etype_edges[0]))
        
        # delete edges
        drop_eids = torch.arange(0,len(etype_edges[0]))[edge_mask==0]
        g_stru = dgl.remove_edges(g_stru, drop_eids, etype=etype)
        
        # add reverse_edges
        g_stru = dgl.add_edges(g_stru, etype_edges[1][drop_eids], etype_edges[0][drop_eids], etype=etype)
    g_stru.ndata['feature'] = g_stru.ndata['feature'].float()
    return g_stru

def GraphAug(g, args):
    if args.augment == 'delete':
        g_attr = generate_stru_graph(g, args)
        g_stru = generate_stru_graph(g, args)
    elif args.augment == 'change':
        g_attr = generate_stru_sign_graph(g, args)
        g_stru = generate_stru_sign_graph(g, args)
    elif args.augment == 'reverse':
        g_attr = generate_stru_status_graph(g, args)
        g_stru = generate_stru_status_graph(g, args)
    elif args.augment == 'composite':
        g_attr = generate_stru_sign_graph(g, args)
        g_stru = generate_stru_graph(g, args)
    return g_attr, g_stru


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
