"""Torch modules for graph convolutions(GCN)."""
# pylint: disable= no-member, arguments-differ, invalid-name
import math

import torch
import torch as th
from torch import nn
from torch.nn import init
from torch.functional import F

from dgl import function as fn
from dgl.base import DGLError
from dgl.utils import expand_as_pair
from dgl.nn import edge_softmax
from dgl.nn import utils


class GraphConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 norm='right',
                 weight=True,
                 bias=True,
                 activation=None,
                 residual=True,
                 allow_zero_in_degree=False):
        super(GraphConv, self).__init__()
#         if norm not in ('none', 'both', 'right'):
        if norm not in ('right'):
            raise DGLError('Invalid norm value. Must be either "none", "both" or "right".'
                           ' But got "{}".'.format(norm))
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree
        self._residual = residual
        
        if self._residual:
            self.loop_weight = nn.Linear(in_feats, out_feats, bias=False)
        
        if weight:
            self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)

        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):

        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)


    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, weight=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            feat_src, feat_dst = expand_as_pair(feat, graph)
            if self._norm == 'both':
                degs = graph.out_degrees().float().clamp(min=1)
                norm = th.pow(degs, -0.5)
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = th.reshape(norm, shp)
                feat_src = feat_src * norm

            if self._residual:
                loop_message = self.loop_weight(feat_dst)
            if graph.num_edges() == 0:
                return loop_message                
                
            if weight is not None:
                if self.weight is not None:
                    raise DGLError('External weight is provided while at the same time the'
                                   ' module has defined its own weight parameter. Please'
                                   ' create the module with flag weight=False.')
            else:
                weight = self.weight
                
            if self._in_feats > self._out_feats:
                # mult W first to reduce the feature size for aggregation.
                if weight is not None:
                    feat_src = th.matmul(feat_src, weight)
                graph.srcdata['h'] = feat_src
                graph.update_all(fn.copy_src(src='h', out='m'),
                                 fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
            else:
                # aggregate first then mult W
                graph.srcdata['h'] = feat_src
                graph.update_all(fn.copy_src(src='h', out='m'),
                                 fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
                if weight is not None:
                    rst = th.matmul(rst, weight)

            if self._norm != 'none':
                degs = graph.in_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = th.reshape(norm, shp)
                rst = rst * norm

            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)
                
            if self._residual:
                rst = rst + loop_message

            return rst

    
    
class GATConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    -1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                    h_dst = h_src[:graph.number_of_dst_nodes()]
            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            # activation
            if self.activation:
                rst = self.activation(rst)
                
            rst = rst.view(len(rst), -1)
                
            return rst
    

    

    
class HeteroGraphConv(nn.Module):
    def __init__(self, mods, dim_values, dim_query, agg_type='attn'):
        super(HeteroGraphConv, self).__init__()
        self.mods = nn.ModuleDict(mods)
        # Do not break if graph has 0-in-degree nodes.
        # Because there is no general rule to add self-loop for heterograph.
        for _, v in self.mods.items():
            set_allow_zero_in_degree_fn = getattr(v, 'set_allow_zero_in_degree', None)
            if callable(set_allow_zero_in_degree_fn):
                set_allow_zero_in_degree_fn(True)
                
        self.dim_values = dim_values
        self.dim_query = dim_query
        
        self.attention = nn.ModuleDict()
        for k, _ in self.mods.items():
            self.attention[k] = nn.Sequential(nn.Linear(dim_values, dim_query), nn.Tanh(), nn.Linear(dim_query, 1, bias=False))
        
        self.agg_type = agg_type
        if agg_type == 'sum':
            self.agg_fn = th.sum
        elif agg_type == 'max':
            self.agg_fn = lambda inputs, dim: th.max(inputs, dim=dim)[0]
        elif agg_type == 'min':
            self.agg_fn = lambda inputs, dim: th.min(inputs, dim=dim)[0]
        elif agg_type == 'stack':
            self.agg_fn = th.stack
        
    def forward(self, g, inputs, mod_args=None, mod_kwargs=None):
        if mod_args is None:
            mod_args = {}
        if mod_kwargs is None:
            mod_kwargs = {}
        outputs = []
        et_scores = []
        et_count = 0
        if isinstance(inputs, tuple) or g.is_block:
            if isinstance(inputs, tuple):
                src_inputs, dst_inputs = inputs
            else:
                src_inputs = inputs
                dst_inputs = inputs[:g.number_of_dst_nodes()]

            for stype, etype, dtype in g.canonical_etypes:
                rel_graph = g[stype, etype, dtype]
                if rel_graph.number_of_edges() == 0:
                    et_scores.append(torch.zeros(g.number_of_dst_nodes(), 1).to(g.device))
                    outputs.append(torch.zeros(g.number_of_dst_nodes(), self.dim_values).to(g.device))
                    continue
                et_count += 1
                dstdata = self.mods[etype](rel_graph, (src_inputs, dst_inputs))
                outputs.append(dstdata)
                et_scores.append(self.attention[etype](dstdata))
        if len(outputs) == 0:
            out_embs = dst_inputs
        else:
            et_dst_data = torch.stack(outputs, dim=0)
            if self.agg_type == 'attn':
                attn = torch.softmax(torch.stack(et_scores, dim=0), dim=0)
                out_embs = (attn * et_dst_data).sum(dim=0)
            elif self.agg_type == 'attn_sum':
                attn = torch.softmax(torch.stack(et_scores, dim=0).mean(dim=1, keepdims=True), dim=0)
                out_embs = (attn * et_dst_data).sum(dim=0)
            elif self.agg_type == 'mean':
                out_embs = torch.sum(torch.stack(et_scores, dim=0), dim=0) / et_count
            else:
                out_embs = self.agg_fn(et_dst_data, dim=0)
        return out_embs, attn    