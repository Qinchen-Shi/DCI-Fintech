import torch.nn as nn
from layers.gat import GAT, Attention
import torch

class U_GCN(nn.Module):
    def __init__(self, nfeat, nclass, nhid1, nhid2, dropout, alpha, nheads):
        super(U_GCN, self).__init__()

        # use GCN or GAT
        self.SGAT1 = GAT(nfeat, nhid1, nhid2, dropout, alpha, nheads)
        self.SGAT2 = GAT(nfeat, nhid1, nhid2, dropout, alpha, nheads)

        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(nhid2, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.attention = Attention(nhid2)
        self.tanh = nn.Tanh()

        self.MLP = nn.Sequential(
            nn.Linear(nhid2, nclass),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x, sadj, sadj2):
        emb1 = self.SGAT1(x, sadj) 
        emb2 = self.SGAT2(x, sadj2)
        emb = torch.stack([emb1, emb2], dim=1)
        emb, att = self.attention(emb)
        return emb # 剩下的等需要了再加