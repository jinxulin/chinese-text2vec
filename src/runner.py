import time
import torch
import numpy as np
from scipy import stats
from sklearn import metrics
from torch.cuda import amp

def cosent_loss(s1_vec, s2_vec, l):
    cosine_sim = torch.cosine_similarity(s1_vec, s2_vec)
    cosine_diff = cosine_sim[None, :] - cosine_sim[:, None]
    labels = l[:, None] > l[None, :]
    labels = labels.long()
    cosine_diff = 20 * cosine_diff - (1 - labels) * 1e12
    cosine_diff = torch.cat((torch.zeros(1).to(cosine_diff.device), cosine_diff.view(-1)), dim=0)
    loss = torch.logsumexp(cosine_diff.view(-1), dim=0)
    return loss

def correct_predictions(s1_vec, s2_vec, label):
    output = torch.cosine_similarity(s1_vec, s2_vec)
    correct = ((output>0.5) == label).sum()
    return correct.item()

def validate(model, dataloader):
    model.eval()
    epoch_start = time.time()
    running_loss = 0.0
    running_accuracy = 0.0
    all_prob = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            if torch.cuda.is_available():
                batch = tuple(t.cuda() for t in batch)
                s1_input_ids, s2_input_ids, label = batch
                with amp.autocast():
                    s1_vec, s2_vec = model(s1_input_ids, s2_input_ids)
                    loss = cosent_loss(s1_vec, s2_vec, label)
            else:
                s1_input_ids, s2_input_ids, label = batch
                s1_vec, s2_vec = model(s1_input_ids, s2_input_ids)
                loss = cosent_loss(s1_vec, s2_vec, label)
            running_loss += loss.item()
            running_accuracy += correct_predictions(s1_vec, s2_vec, label)
            all_prob.extend(torch.cosine_similarity(s1_vec, s2_vec).detach().cpu().numpy())
            all_labels.extend(label.detach().cpu().numpy())
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_accuracy / (len(dataloader.dataset))
    auc = metrics.roc_auc_score(all_labels, all_prob)
    pearsonr = stats.pearsonr(all_labels, all_prob)[0]
    return epoch_time, epoch_loss, epoch_accuracy, auc, pearsonr
