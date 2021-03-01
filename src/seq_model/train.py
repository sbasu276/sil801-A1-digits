import sys
import argparse
import os
from itertools import groupby
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import torchvision.transforms.functional as TF
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader 
from dataloader import DigitsDataset
from model import DigitsNet
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def parse():
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('--img_dir', dest="img_dir", default="data/seq_imgs")
    parser.add_argument('--labels', dest="labels", default="data/seq_labels.txt")
    parser.add_argument('--batch_size', dest="batch_size", default=64, type=int)
    parser.add_argument('--set', dest="set", default=4, type=int)
    parser.add_argument('--epochs', dest="epochs", default=30, type=int)
    args = parser.parse_args()
    return args

def main(args):
    epochs = args.epochs
    num_classes = 12
    blank_label = 11
    cnn_output_height = 5 #4
    cnn_output_width = 77 #32
    digits_per_sequence = 10
    batch_size=args.batch_size
    set_num = args.set

    img_transforms = T.Compose([T.ToTensor()])
    d = {}
    with open(args.labels,"r") as f:
        for e in f.readlines():
            k, v = e.strip().split(",")
            d[k] = v
    train_set = []
    with open("data/train_%s.txt"%set_num, "r") as f:
        for e in f.readlines():
            train_set.append(e.strip())
    test_set = []
    with open("data/test_%s.txt"%set_num, "r") as f:
        for e in f.readlines():
            test_set.append(e.strip())

    train_dataset = DigitsDataset(args.img_dir, d, train_set, transforms = img_transforms)
    test_dataset = DigitsDataset(args.img_dir, d, test_set, transforms = img_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)#, collate_fn=utils.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1)#, collate_fn=utils.collate_fn)
    
    model = DigitsNet().to(device)
    criterion = nn.CTCLoss(blank=blank_label, reduction='mean', zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    max_acc = 0
    for ep in range(epochs):
    # ============================================ TRAINING ============================================================
        train_correct = 0
        train_total = 0
        for x_train, y_train, _ in train_loader:
            batch_size = x_train.shape[0]  # x_train.shape == torch.Size([64, 28, 140])
            #x_train = x_train.view(x_train.shape[0], 1, x_train.shape[1], x_train.shape[2])
            optimizer.zero_grad()
            y_pred = model(x_train.cuda())
            y_pred = y_pred.permute(1, 0, 2)  # y_pred.shape == torch.Size([64, 32, 11])
            input_lengths = torch.IntTensor(batch_size).fill_(cnn_output_width)
            target_lengths = torch.IntTensor([len(t) for t in y_train])
            loss = criterion(y_pred, y_train, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()
            _, max_index = torch.max(y_pred, dim=2)  # max_index.shape == torch.Size([32, 64])
            for i in range(batch_size):
                raw_prediction = list(max_index[:, i].detach().cpu().numpy())  # len(raw_prediction) == 32
                prediction = torch.IntTensor([c for c, _ in groupby(raw_prediction) if c != blank_label])
                if len(prediction) == len(y_train[i]) and torch.all(prediction.eq(y_train[i])):
                    train_correct += 1
                train_total += 1
        print('Epoch: ', ep,'TRAINING. Correct: ', train_correct, '/', train_total, '=', train_correct / train_total)

        # ============================================ VALIDATION ==========================================================
        val_correct = 0
        val_total = 0
        for x_val, y_val, _ in test_loader:
            batch_size = x_val.shape[0]
            #x_val = x_val.view(x_val.shape[0], 1, x_val.shape[1], x_val.shape[2])
            y_pred = model(x_val.cuda())
            y_pred = y_pred.permute(1, 0, 2)
            input_lengths = torch.IntTensor(batch_size).fill_(cnn_output_width)
            target_lengths = torch.IntTensor([len(t) for t in y_val])
            criterion(y_pred, y_val, input_lengths, target_lengths)
            _, max_index = torch.max(y_pred, dim=2)
            for i in range(batch_size):
                raw_prediction = list(max_index[:, i].detach().cpu().numpy())
                prediction = torch.IntTensor([c for c, _ in groupby(raw_prediction) if c != blank_label])
                if len(prediction) == len(y_val[i]) and torch.all(prediction.eq(y_val[i])):
                    val_correct += 1
                val_total += 1
        
        if val_correct/val_total > max_acc:
            max_acc = val_correct/val_total
            torch.save(model.state_dict(), "digits-net.pth")

    print('\t Split-%s Max Acc: %s'%(set_num, max_acc))

if __name__ == "__main__":
    args = parse()
    main(args)
