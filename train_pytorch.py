import torch
import math
import numpy as np
from DatasetIO import FbankDataset, validation_split
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from pathlib import Path

nBatch = 512
dtype = torch.FloatTensor

print("Loading Datasets...")
Dtrain = FbankDataset()
train_ds, valid_ds = validation_split(Dtrain,val_share=0.1)
train_loader = DataLoader(Dtrain, batch_size=nBatch, num_workers=6,pin_memory=True)
valid_loader = DataLoader(valid_ds, batch_size=len(valid_ds), num_workers=6, pin_memory=True)
Dtest = FbankDataset(is_test=True)
test_loader = DataLoader(Dtest, batch_size=nBatch)

D_in = Dtrain[0]['x'].shape[0]
D_out = 1943

# model
#======================================================
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dp1 = nn.Dropout(p=0.1)
        self.fc1 = nn.Linear(D_in,2048)
        self.bn1 = nn.BatchNorm1d(2048)

        self.dp2 = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(2048,2048)
        self.bn2 = nn.BatchNorm1d(2048)

        self.dp3 = nn.Dropout(p=0.1)
        self.fc3 = nn.Linear(2048,2048)
        self.bn3 = nn.BatchNorm1d(2048)

        self.dp4 = nn.Dropout(p=0.1)
        self.fc4 = nn.Linear(2048,2048)
        self.bn4 = nn.BatchNorm1d(2048)

        self.dp5 = nn.Dropout(p=0.1)
        self.fc5 = nn.Linear(2048,D_out)
        self.bn5 = nn.BatchNorm1d(D_out)
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(self.dp1(x))))
        x = F.relu(self.bn2(self.fc2(self.dp2(x))))
        x = F.relu(self.bn3(self.fc3(self.dp3(x))))
        x = F.relu(self.bn4(self.fc4(self.dp4(x))))
        return self.bn5(self.fc5(self.dp5(x)))
net = Net()
net.cuda()
#======================================================

exp_name = "b_bn_relu_dp_0.1"

csv_save = exp_name + ".csv"
model_save = exp_name + ".pt"

model_file = Path(model_save)
if model_file.is_file():
    print("Resuming model in file {}".format(model_save))
    net.load_state_dict(torch.load(model_save))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr=1e-4,betas=(0.5, 0.999))


def count_correct(targ, pred):
    """
        Input: 
            targ, label of training data
            pred, probability prediction
    """
    target = targ.numpy()
    pred_y = np.argmax(pred.data.cpu().numpy(), axis=1)
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

def train(model, to_valid=False):
    """
        Main training loop
    """
    batch_num = len(train_loader)
    print("Total Batch number {}".format(batch_num))
    correct_ctr, valid_ctr = 0, 0
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
            correct_ctr += count_correct(train_batch['y'], prob_y)
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
        print("Correct Count: {}/{} ({}%),".format( correct_ctr, valid_ctr, correct_ctr/valid_ctr*100))

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