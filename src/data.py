import torch
from hanziconv import HanziConv
from torch.utils.data import Dataset

class SentencePairDataset(Dataset):
    def __init__(self,bert_tokenizer, df, max_len=64):
        self.bert_tokenizer = bert_tokenizer
        self.max_seq_len = max_len
        self.seqs1, self.seqs2, self.labels = self.get_input(df)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.seqs1[idx], self.seqs2[idx], self.labels[idx]

    def get_input(self, df):
        sentences_1 = map(HanziConv.toSimplified, df['sentence1'].values)
        sentences_2 = map(HanziConv.toSimplified, df['sentence2'].values)
        tokens_seq_1 = list(map(self.bert_tokenizer.encode, sentences_1))
        tokens_seq_2 = list(map(self.bert_tokenizer.encode, sentences_2))
        result = list(map(self.trunate_and_pad, tokens_seq_1, tokens_seq_2))
        seqs1 = [i[0] for i in result]
        seqs2 = [i[1] for i in result]
        labels = df['label'].values
        return torch.Tensor(seqs1).type(torch.long), torch.Tensor(seqs2).type(torch.long), torch.Tensor(labels).type(torch.float)

    def trunate_and_pad(self, seq1, seq2):
        max_len = self.max_seq_len
        if len(seq1) > max_len:
            seq1 = seq1[:max_len]
        else:
            seq1 = seq1 + (max_len - len(seq1)) * [0]
        if len(seq2) > max_len:
            seq2 = seq2[:max_len]
        else:
            seq2 = seq2 + (max_len - len(seq2)) * [0]
        return seq1, seq2
