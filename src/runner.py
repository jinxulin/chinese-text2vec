import time
import torch
import numpy as np
from scipy import stats
from sklearn import metrics
from tqdm import tqdm
from torch.cuda import amp


def correct_predictions(s1_vec, s2_vec, label):
    output = torch.cosine_similarity(s1_vec, s2_vec)
    correct = ((output>0.5) == label).sum()
    return correct.item()


def train(model, loss_func, dataloader, optimizer, args):
    start_time = time.time()
    running_loss = 0
    batch_time_avg = 0
    tqdm_batch_iterator = tqdm(dataloader)
    for batch_index, batch in enumerate(tqdm_batch_iterator):
        model.zero_grad()
        if args.device == 'cuda':
            batch = tuple(t.cuda() for t in batch)
        s1_input_ids, s2_input_ids, label = batch
        if args.device == 'cuda' and args.enable_amp:
            scaler = amp.GradScaler()
            with amp.autocast():
                s1_vec, s2_vec = model(s1_input_ids, s2_input_ids)
                loss = loss_func(s1_vec, s2_vec, label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            s1_vec, s2_vec = model(s1_input_ids, s2_input_ids)
            loss = loss_func(s1_vec, s2_vec, label)
            loss.backward()
            optimizer.step()
        running_loss += loss.item()
        batch_time_avg = time.time() - start_time
        description = "Running metrics on average. time: {:.4f}s, loss: {:.4f}".format(batch_time_avg / (batch_index + 1), running_loss / (batch_index + 1))
        tqdm_batch_iterator.set_description(description)


def validate(model, loss_func, dataloader, args):
    model.eval()
    epoch_start = time.time()
    running_loss = 0.0
    running_accuracy = 0.0
    all_prob = []
    all_labels = []
    tqdm_batch_iterator = tqdm(dataloader)
    with torch.no_grad():
        for batch in tqdm_batch_iterator:
            if args.device == 'cuda':
                batch = tuple(t.cuda() for t in batch)
            s1_input_ids, s2_input_ids, label = batch
            if args.device == 'cuda' and args.enable_amp:
                with amp.autocast():
                    s1_vec, s2_vec = model(s1_input_ids, s2_input_ids)
                    loss = loss_func(s1_vec, s2_vec, label)
            else:
                s1_vec, s2_vec = model(s1_input_ids, s2_input_ids)
                loss = loss_func(s1_vec, s2_vec, label)
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
