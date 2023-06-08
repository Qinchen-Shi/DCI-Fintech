import torch.nn as nn
from layers import GraphCNN, MLP
import torch.nn.functional as F
import sys
sys.path.append("models/")

# 只用在了main.py里的finetune函数里，只是为了算AUC值，可以理解成
class Classifier(nn.Module):    # 因为继承了NN所以很多东西不用写在这里
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, final_dropout, neighbor_pooling_type, device):
        super(Classifier, self).__init__()
        self.gin = GraphCNN(num_layers, num_mlp_layers, input_dim, hidden_dim, neighbor_pooling_type, device)
        self.linear_prediction = nn.Linear(hidden_dim, 1)
        self.final_dropout = final_dropout
        
    def forward(self, seq1, adj):
        h_1 = self.gin(seq1, adj)   # 先是生成features embedding
        score_final_layer = F.dropout(self.linear_prediction(h_1), self.final_dropout, training = self.training)    # 分别是张量，dropout概率和是不是training set
        return score_final_layer    # 返回的是失活张量