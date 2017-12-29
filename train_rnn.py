import torch
import time
import math
import os
import numpy as np
from DatasetIO import SequenceDataset
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from model import Lstm

from pathlib import Path

import argparse

parser = argparse.ArgumentParser(description='Train MINST dataset')

parser.add_argument('--resume', dest='resume', action='store_true',
                    help='resume previous model')

parser.add_argument('--name', dest='name', action='store',default='exp1',
                    help='name of the experiment')

parser.add_argument('--cuda', dest='cuda', action='store_true',
                    help='using gpu as device')

args = parser.parse_args()
print(args)

dtype = torch.FloatTensor

print("Loading Datasets...")
Dtrain = SequenceDataset()
Dtest = SequenceDataset(is_test=True)

D_in = Dtrain[0]['x'].shape[1]
D_out = 1943

train_loader = DataLoader(Dtrain, batch_size=1, num_workers=4,pin_memory=args.cuda)
test_loader = DataLoader(Dtest, batch_size=1)

# model
#======================================================
net = Lstm(D_in, 1024, D_out,dropout=0.1)
if args.cuda:
    net.cuda()
#======================================================


exp_name = args.name + "/"

if not Path(exp_name).is_dir():
    os.mkdir(exp_name)

csv_save = exp_name + "result.csv"
model_save = exp_name + "model.pt"

model_file = Path(model_save)
if model_file.is_file() and args.resume:
    print("Resuming model in file {}".format(model_save))
    net.load_state_dict(torch.load(model_save))

def criterion(prob_y, target_y):
    fn = nn.CrossEntropyLoss()
    return fn(prob_y, target_y)

optimizer = optim.Adam(net.parameters(),lr=1e-4,betas=(0.5, 0.999))


def count_correct(pred, target):
    """
        Input: 
            targ, label of training data
            pred, probability prediction
    """
    pred_y = np.argmax(pred.data.cpu().numpy(), axis=1)
    return (pred_y==target).sum()

def mapping(a):
    return Dtrain.state_39[a]

def phone_count(pred, target):
    """
        Input: 
            targ, label of training data
            pred, probability prediction
    """
    vfunc = np.vectorize(mapping)
    target = vfunc(target)
    pred_y = np.argmax(pred.data.cpu().numpy(), axis=1)
    pred_y = vfunc(pred_y)
    return (pred_y==target).sum()

def test(model, ds):
    mapping = ds.ctr_phone
    with open(csv_save, 'w') as f:
        print('Writing to file...'+csv_save)
        f.write('ID,prediction\n')
        for i_batch, test_batch in enumerate(Dtest):
            shape = test_batch['x'].shape
            x = Variable(torch.FloatTensor(test_batch['x']).view(shape[0],1,shape[1])).type(dtype)
            model.hidden = model.init_hidden()
            if args.cuda:
                x = x.cuda()
                model.hidden = model.hidden[0].cuda(), model.hidden[1].cuda()
            
            model.eval()
            prob_y = torch.squeeze(model(x))
            if args.cuda:
                prob_y = prob_y.data.cpu().numpy()
            else:
                prob_y = prob_y.data.numpy()
            pred_y = np.argmax(prob_y, axis=1)
            for i in range(len(pred_y)):
                f.write(test_batch['label']+'_'+str(i+1) + ',' + ds.state_39[pred_y[i]]+'\n')

def train(model,to_valid=False):
    """
        Main training loop
    """
    batch_num = len(train_loader)
    print("Total Seq number {}".format(batch_num))
    correct_ctr, valid_ctr, real_ctr = 0, 0, 0
    running_loss = Variable(torch.FloatTensor(1).zero_(),volatile=True)
    running_loss_ctr=0
    valid_loss = Variable(torch.FloatTensor(1).zero_(),volatile=True)
    flag_first=False

    if args.cuda:
        running_loss, valid_loss = running_loss.cuda(), valid_loss.cuda()
    st = time.time()
    for idx, seq in enumerate(Dtrain):
        shape = seq['x'].shape
        x = Variable(torch.FloatTensor(seq['x']).view(shape[0],1,shape[1]))
        target = Variable(torch.LongTensor(seq['y']))
        model.hidden = model.init_hidden()

        optimizer.zero_grad()
        if args.cuda:
            x, target = x.cuda(), target.cuda()
            model.hidden = model.hidden[0].cuda(), model.hidden[1].cuda()
            

        if idx > len(Dtrain) * 0.9:
            # Validating
            if not to_valid:
                continue
            if not flag_first:
                flag_first = True
                print("Validating")
            model.eval()
            prob_y = torch.squeeze(model(x))
            loss = criterion(prob_y, target)

            valid_loss += loss
            correct_ctr += count_correct(prob_y, seq['y'])
            real_ctr += phone_count(prob_y, seq['y'])
            valid_ctr += seq['y'].shape[0]

        else:
            # Training
            model.train()
            prob_y = torch.squeeze(model(x))
            loss = criterion(prob_y, target)
            loss.backward()
            optimizer.step()

            running_loss += loss
            running_loss_ctr += seq['y'].shape[0]
            if(idx % 400 == 399):
                print("({}) seq #{}, loss:{}".format(timeSince(st),idx, running_loss.data[0]/running_loss_ctr))

    if to_valid:
        print("Correct Count: {}/{} ({}%), Correct label: {}/{} ({}%)".format( correct_ctr, valid_ctr, correct_ctr/valid_ctr*100, real_ctr, valid_ctr, real_ctr/valid_ctr*100))

    print("Avg. loss:{}, Valid loss:{}".format(running_loss.data[0]/running_loss_ctr, valid_loss.data[0]/valid_ctr if valid_ctr!=0 else "Nan"))

def post_training():
    #print("Start Validation")
    #validation()
    print("Start Predicting...")
    print("Calculating {} elements".format(len(Dtest)))
    test(net, Dtest)
    torch.save(net.state_dict(), model_save)

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

print("The dataset dimension is ({} -> {})".format(D_in,D_out))
print("Running experiment [{}]".format(exp_name))
print(net)
print("Start Training...")

try:
    start = time.time()
    for epoch in range(200):
        print("epoch #{} ({})".format(epoch, timeSince(start)))
        train(net, to_valid=True)

except KeyboardInterrupt:
    post_training()
    exit()

post_training()