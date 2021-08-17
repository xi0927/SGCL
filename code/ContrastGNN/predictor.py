import torch
from torch import nn
from torch.nn import init
from torch.functional import F
from dotmap import DotMap
from sklearn import metrics


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