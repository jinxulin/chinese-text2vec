import time
import torch
from torch import nn
from torch.cuda import amp
import torch.nn.functional as F

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


class SBERTLoss(nn.Module):
    def __init__(self, dim):
        super(SBERTLoss, self).__init__()
        self.fc = nn.Linear(dim * 3, 1, bias=False)

    def forward(self, s1_vec, s2_vec, l):
        inputs = torch.cat([s1_vec, s2_vec, torch.abs(s1_vec - s2_vec)], dim=1)
        logits = self.fc(inputs).view(-1)
        loss = F.binary_cross_entropy_with_logits(logits, l)
        return loss


class CoSENTLoss(nn.Module):
    def __init__(self):
        super(CoSENTLoss, self).__init__()

    def forward(self, s1_vec, s2_vec, l):
        cosine_sim = torch.cosine_similarity(s1_vec, s2_vec)
        cosine_diff = cosine_sim[None, :] - cosine_sim[:, None]
        labels = l[:, None] > l[None, :]
        labels = labels.long()
        cosine_diff = 20 * cosine_diff - (1 - labels) * 1e12
        cosine_diff = torch.cat((torch.zeros(1).to(cosine_diff.device), cosine_diff.view(-1)), dim=0)
        loss = torch.logsumexp(cosine_diff.view(-1), dim=0)
        return loss
