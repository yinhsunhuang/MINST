import torch
import math
import numpy as np
from DatasetIO import FbankDataset, validation_split
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import Layer
from Network import Network

nBatch = 256
dtype = torch.FloatTensor
# Adam parameters
# learning rate
alpha = 0
alpha_max = 1e-4
alpha_min = 0
alpha_Tmul = 2
alpha_Ti = 10
alpha_T = 0
beta1 = 0.9
beta1_t = beta1
beta2 = 0.999
beta2_t = beta2
epsilon = 1e-8
# -----------
l2_lambda = 0

# [warm]_[act_fun]_[init]_[dp_dpProb]_[#].csv
filename_save = "warm_relu_deep_state_dp_0.1.csv"

# def criterion(targ, pred):
#     idx = targ.data
#     print(-pred[idx])
#     return (-pred[idx]).sum()
criterion = nn.CrossEntropyLoss()

def count_correct(targ, pred):
    """
        Input: 
            targ, label of training data
            pred, probability prediction
    """
    target = targ.numpy()
    pred_y = np.argmax(pred.data.cpu().numpy(), axis=1)
    return (pred_y==target).sum()

def test(ds):
    mapping = ds.ctr_phone
    with open(filename_save, 'w') as f:
        print('Writing to file...'+filename_save)
        f.write('ID,prediction\n')
        for i_batch, test_batch in enumerate(test_loader):
            x = Variable(test_batch['x'].type(dtype)).cuda()
            prob_y = model.out(x,is_test=True)
            prob_y = prob_y.data.cpu().numpy()
            pred_y = np.argmax(prob_y, axis=1)
            for i in range(len(pred_y)):
                f.write(test_batch['label'][i] + ',' + ds.state_39[pred_y[i]]+'\n')
                #f.write(test_batch['label'][i] + ',' + ds.ctr_phone[pred_y[i]]+'\n')

def grad_update(parameters):
    """
        Update parameters with Adam method
    """
    global param_m_grad,param_v_grad, beta1, beta1_t, beta2, beta2_t, grad_t,alpha
    for i, param in enumerate(parameters):
        param_m_grad[i] = beta1 * param_m_grad[i] + (1-beta1) * param.grad.data
        param_v_grad[i] = beta2 * param_v_grad[i] + (1-beta2) * param.grad.data.pow(2)
        est_m = param_m_grad[i]/(1-beta1_t)
        est_v = param_v_grad[i]/(1-beta2_t)
        est_v = est_v.sqrt()
        param.data -= alpha * est_m/(est_v + epsilon)
        if beta1_t < 1e-14:
            beta1_t = 0
        else:
            beta1_t *= beta1
        if beta2_t < 1e-14:
            beta2_t = 0
        else:
            beta2_t *= beta2

    for param in parameters:
        param.grad.data.zero_()

def train(to_valid=False):
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
        target = Variable(train_batch['y'].type(torch.LongTensor)).cuda()
        if i_batch > batch_num * 0.9:
            if not flag_first:
                flag_first = True
                print("Validating")
            # Validating
            if not to_valid:
                break
            prob_y = model.out(x,is_test=True)

            loss = criterion(prob_y,target)
            for p in model.parameters():
                loss += l2_lambda * p.norm(2)
            valid_loss += loss
            correct_ctr += count_correct(train_batch['y'], prob_y)
            valid_ctr += train_batch['y'].size()[0]
        else:
            # Training
            prob_y = model.out(x,is_test=False)
            loss = criterion(prob_y,target)
            for p in model.parameters():
                loss += l2_lambda * p.norm(2)
            running_loss += loss
            running_loss_ctr += train_batch['y'].size()[0]
            if(i_batch % 400 == 399):
                print("batch #{}, loss:{}".format(i_batch, running_loss.data[0]/running_loss_ctr))
                
            loss.backward()
            grad_update(parameters)
    if to_valid:
        print("Correct Count: {}/{} ({}%),".format( correct_ctr, valid_ctr, correct_ctr/valid_ctr*100))
    print("Avg. loss:{}, Valid loss:{}".format(running_loss.data[0]/running_loss_ctr, valid_loss.data[0]/valid_ctr))

print("Loading Datasets...")
Dtrain = FbankDataset()
train_ds, valid_ds = validation_split(Dtrain,val_share=0.1)
train_loader = DataLoader(Dtrain, batch_size=nBatch, num_workers=4,pin_memory=True)
valid_loader = DataLoader(valid_ds, batch_size=len(valid_ds), num_workers=4, pin_memory=True)
Dtest = FbankDataset(is_test=True)
test_loader = DataLoader(Dtest, batch_size=nBatch)

D_in = Dtrain[0]['x'].shape[0]
D_out = Dtrain.y_dim

print("The dataset dimension is ({} -> {})".format(D_in,D_out)) 
print(filename_save)
model = Network()
# initialization of network parameters
hidden_layer_num = [D_in, 1024,1024,1024,1024, D_out]
for i in range(len(hidden_layer_num)-1):
    in_dim = hidden_layer_num[i]
    out_dim = hidden_layer_num[i+1]
    W = Variable( (0.2*(0.5-torch.rand(in_dim, out_dim))).type(dtype).cuda(), requires_grad=True)
    #W = Variable( torch.normal(torch.zeros(in_dim, out_dim), 0.1*torch.ones(in_dim, out_dim) ).type(dtype).cuda(), requires_grad=True)
    #b = Variable( torch.normal(torch.zeros(1, out_dim), 0.1*torch.ones(1, out_dim) ).type(dtype).cuda(), requires_grad=True)
    b = Variable( (0.2*(0.5-torch.rand(1, out_dim))).type(dtype).cuda(), requires_grad=True)
    if i < len(hidden_layer_num)-2:
        ll = Layer.FCLayer(W,b,act_fnc=F.relu)
    else:
        ll = Layer.FCLayer(W,b,act_fnc=None)
    model.append(Layer.DropoutLayer(dropout=0.1)) 
    model.append(ll)

model.describe()
print("L2_lr:{}".format(l2_lambda))

parameters = model.parameters()

param_m_grad = [
   torch.zeros(parm.data.size()).type(dtype).cuda() for parm in model.parameters()
] # 1st moment vector

param_v_grad = [
   torch.zeros(parm.data.size()).type(dtype).cuda() for parm in model.parameters()
] # 2nd moment vector

print("Start Training...")

try:
    for epoch in range(200):
        alpha = np.float32(alpha_min + 0.5*(alpha_max-alpha_min)*(1+np.cos(alpha_T/alpha_Ti*math.pi)))
        #alpha = 1e-5
        print("epoch #{}".format(epoch))
        print("learning_rate: {}".format(alpha))
        train(to_valid=True)
        if alpha_T == alpha_Ti-1:
            alpha_Ti *= alpha_Tmul
            alpha_T = 0
        else:
            alpha_T+=1

except KeyboardInterrupt:
    print("Interrupted...")

#print("Start Validation")
#validation()
print("Start Predicting...")
print("Calculating {} elements".format(len(Dtest)))
test(Dtest)
