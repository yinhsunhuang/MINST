import torch
import math
import os
import numpy as np
from DatasetIO import FbankDataset, validation_split
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from model import Net

from pathlib import Path

import argparse

parser = argparse.ArgumentParser(description='Train MINST dataset')

parser.add_argument('--resume', dest='resume', action='store_true',
                    help='resume previous model')

parser.add_argument('--name', dest='name', action='store',default='exp1',
                    help='name of the experiment')

args = parser.parse_args()


nBatch = 512
dtype = torch.FloatTensor

print("Loading Datasets...")
Dtrain = FbankDataset()
train_ds, valid_ds = validation_split(Dtrain,val_share=0.1)
train_loader = DataLoader(Dtrain, batch_size=nBatch, num_workers=6,pin_memory=True)
valid_loader = DataLoader(valid_ds, batch_size=len(valid_ds), num_workers=4, pin_memory=True)
Dtest = FbankDataset(is_test=True)
test_loader = DataLoader(Dtest, batch_size=nBatch)

D_in = Dtrain[0]['x'].shape[0]
D_out = 1943

# model
#======================================================
net = Net(D_in, D_out)
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


def count_correct(pred, targ):
    """
        Input: 
            targ, label of training data
            pred, probability prediction
    """
    target = targ.numpy()
    pred_y = np.argmax(pred.data.cpu().numpy(), axis=1)
    return (pred_y==target).sum()

def mapping(a):
    return Dtrain.state_39[a]

def phone_count(pred, targ):
    """
        Input: 
            targ, label of training data
            pred, probability prediction
    """
    vfunc = np.vectorize(mapping)
    target = targ.numpy()
    target = vfunc(target)
    pred_y = np.argmax(pred.data.cpu().numpy(), axis=1)
    pred_y = vfunc(pred_y)
    return (pred_y==target).sum()

def test(model, ds):
    mapping = ds.ctr_phone
    with open(csv_save, 'w') as f:
        print('Writing to file...'+csv_save)
        f.write('ID,prediction\n')
        for i_batch, test_batch in enumerate(test_loader):
            x = Variable(test_batch['x'].type(dtype)).cuda()
            model.eval()
            prob_y = model(x)
            prob_y = prob_y.data.cpu().numpy()
            pred_y = np.argmax(prob_y, axis=1)
            for i in range(len(pred_y)):
                f.write(test_batch['label'][i] + ',' + ds.state_39[pred_y[i]]+'\n')

def train(model,to_valid=False):
    """
        Main training loop
    """
    batch_num = len(train_loader)
    print("Total Batch number {}".format(batch_num))
    correct_ctr, valid_ctr, real_ctr = 0, 0, 0
    running_loss = Variable(torch.FloatTensor(1).zero_().cuda(),volatile=True)
    running_loss_ctr=0
    valid_loss = Variable(torch.FloatTensor(1).zero_().cuda(),volatile=True)
    flag_first=False
    for i_batch, train_batch in enumerate(train_loader):
        x = Variable(train_batch['x']).type(dtype).cuda()
        target = Variable(train_batch['y']).type(torch.LongTensor).cuda()

        optimizer.zero_grad()

        if i_batch > batch_num * 0.9:
            # Validating
            if not to_valid:
                continue
            if not flag_first:
                flag_first = True
                print("Validating")
            model.eval()
            prob_y = model(x)
            loss = criterion(prob_y, target)

            valid_loss += loss
            correct_ctr += count_correct(prob_y, train_batch['y'])
            real_ctr += phone_count(prob_y, train_batch['y'])
            valid_ctr += train_batch['y'].size()[0]

        else:
            # Training
            model.train()
            prob_y = model(x)
            loss = criterion(prob_y, target)
            loss.backward()
            optimizer.step()

            running_loss += loss
            running_loss_ctr += train_batch['y'].size()[0]
            if(i_batch % 400 == 399):
                print("batch #{}, loss:{}".format(i_batch, running_loss.data[0]/running_loss_ctr))

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


print("The dataset dimension is ({} -> {})".format(D_in,D_out))
print("Running experiment [{}]".format(exp_name))
print(net)
print("Start Training...")

try:
    for epoch in range(200):
        print("epoch #{}".format(epoch))
        train(net, to_valid=True)

except KeyboardInterrupt:
    post_training()
    exit()

post_training()