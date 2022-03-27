import argparse
import os
import pandas as pd
import torch
from datetime import datetime
from data import SentencePairDataset
from model import *
from runner import *
from torch.utils.data import DataLoader
from transformers import BertConfig, BertModel, BertTokenizer


def set_args():
    parser = argparse.ArgumentParser('--使用transformers实现cosent')
    parser.add_argument('--train_data_path', default='../data/train.tsv', type=str, help='训练数据集')
    parser.add_argument('--dev_data_path', default='../data/dev.tsv', type=str, help='测试数据集')
    parser.add_argument('--pretrain_dir', default='../pretrain/', type=str, help='预训练模型模型位置')
    parser.add_argument('--train_batch_size', default=16, type=int, help='训练批次的大小')
    parser.add_argument('--dev_batch_size', default=16, type=int, help='验证批次的大小')
    parser.add_argument('--output_dir', default='../output/', type=str, help='模型输出目录')
    parser.add_argument('--num_epochs', default=10, type=int, help='训练几轮')
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='学习率大小')
    parser.add_argument('--truncate_len', default=32, type=int, help='文本截断的长度')
    parser.add_argument('--model', default='cosent', type=str, help='模型名称，可以是cosent或sbert')
    parser.add_argument('--device', default='cuda', type=str, help='设备选择, cpu or cuda')
    parser.add_argument('--enable_amp', default=True, type=bool, help='启用混合精度加速')
    return parser.parse_args()

def run(args):
    # 预训练模型加载
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    tokenizer = BertTokenizer.from_pretrained(os.path.join(args.pretrain_dir, 'vocabs.txt'), local_files_only=True, do_lower_case=True)
    pretrain_config = BertConfig.from_pretrained(os.path.join(args.pretrain_dir, 'config.json'))
    pretrain_model = BertModel.from_pretrained(args.pretrain_dir, config=pretrain_config)
    # 模型初始化
    model = Model(pretrain_model)
    model.to(device)

    # 损失函数初始化
    if args.model == 'cosent':
        loss_func = CoSENTLoss().to(device)
    elif args.model == 'sbert':
        loss_func = SBERTLoss(pretrain_config.hidden_size).to(device)

    # 数据加载
    df_train = pd.read_csv(args.train_data_path, sep='\t')
    df_dev = pd.read_csv(args.dev_data_path, sep='\t')
    train_data = SentencePairDataset(tokenizer, df_train, max_len=args.truncate_len)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=args.train_batch_size)
    dev_data = SentencePairDataset(tokenizer, df_dev, max_len=args.truncate_len)
    dev_loader = DataLoader(dev_data, shuffle=False, batch_size=args.dev_batch_size)

    # 优化器设置
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{
        'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay':0.01
    },{
        'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay':0.0
    }]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    for epoch in range(args.num_epochs):
        print("* Train for epoch {}:".format(epoch))
        train(model, loss_func, train_loader, optimizer, args)
        print("* Validation for epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy, epoch_auc, epoch_pearsonr = validate(model, loss_func, dev_loader, args)
        result_info = "Valid metrics. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%, auc: {:.4f}, pearsonr: {:.4f}\n".format(epoch_time, epoch_loss, (epoch_accuracy * 100), epoch_auc, epoch_pearsonr)
        print(result_info)
        torch.save({"epoch": epoch,
                    "model": model.state_dict(),
                    "valid_losses": epoch_loss},
                    os.path.join(args.output_dir, "model_{0}.pth.tar".format(epoch)))
        with open("{0}/history.txt".format(args.output_dir), "a") as history:
                history.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + f" epoch {epoch}: " + result_info)

if __name__ == "__main__":
    # 参数设置
    args = set_args()
    run(args)
