"""Torch modules for graph convolutions(GCN)."""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch
from torch import nn
from torch.nn import init
from torch.functional import F

import dgl
from dgl import function as fn
from dgl.base import DGLError
from dgl.utils import expand_as_pair
from dgl.nn import edge_softmax
from dgl.nn import utils

from dotmap import DotMap
from my_nn import GraphConv, GATConv, HeteroGraphConv
from predictor import *
import numpy as np

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
