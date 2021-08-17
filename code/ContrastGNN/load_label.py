import torch

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
