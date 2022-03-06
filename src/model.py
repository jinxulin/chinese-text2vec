import time
import torch
from torch import nn
from torch.cuda import amp

class Model(nn.Module):
    def __init__(self, bert):
        super(Model, self).__init__()
        self.bert = bert

    def forward(self, s1_input_ids, s2_input_ids):
        s1_mask = torch.ne(s1_input_ids, 0)
        s2_mask = torch.ne(s2_input_ids, 0)
        s1_output = self.bert(input_ids=s1_input_ids, attention_mask=s1_mask)
        s2_output = self.bert(input_ids=s2_input_ids, attention_mask=s2_mask)
        last_hidden1 = s1_output.last_hidden_state
        last_hidden2 = s2_output.last_hidden_state
        seq_length = last_hidden1.size(1)
        s1_vec = torch.avg_pool1d(last_hidden1.transpose(1, 2), kernel_size=seq_length).squeeze(-1)
        s2_vec = torch.avg_pool1d(last_hidden2.transpose(1, 2), kernel_size=seq_length).squeeze(-1)
        return s1_vec, s2_vec
