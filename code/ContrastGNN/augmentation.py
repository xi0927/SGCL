import numpy as np
import torch
import dgl
import random

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