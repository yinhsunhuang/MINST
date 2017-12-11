import torch
import numpy as np
from FbankDataset import FbankDataset, validation_split
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import Layer
from Network import Network

nBatch = 256
dtype = torch.FloatTensor
# Adam parameters
alpha = 1e-5
beta1 = 0.5
beta2 = 0.999
epsilon = 1e-8
grad_t = 0
# -----------
filename_save = "relu_dropout_0.5.csv"

def criterion(targ, pred):
    return (-targ * pred.log()).sum()

def count_correct(targ, pred):
    """
        Input: 
            targ, label of training data
            pred, probability prediction
    """
    target = np.argmax(targ.numpy(), axis=1)
    pred_y = np.argmax(pred.data.cpu().numpy(), axis=1)
    return (pred_y==target).sum()

def validation():
    """
        Validation with valid_loader, not used in the new version
    """
    for valid_batch in valid_loader:
        x = Variable(valid_batch['x'].type(dtype)).cuda()
        prob_y = logistic_reg(x,W1,b1,W2,b2,W3,b3,W4,b4,W5,b5)
        correct_count = count_correct(valid_batch['y'], prob_y)
        print("Correct Count: {} ({}%),".format( correct_count, correct_count/prob_y.data.size[0]*100))

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

def grad_update(parameters):
    """
        Update parameters with Adam method
    """
    global param_m_grad,param_v_grad, beta1, beta1, grad_t
    grad_t += 1
    for i, param in enumerate(parameters):
        param_m_grad[i] = beta1 * param_m_grad[i] + (1-beta1) * param.grad.data
        param_v_grad[i] = beta2 * param_v_grad[i] + (1-beta2) * param.grad.data.pow(2)
        param.data -= alpha * ( param_m_grad[i]/(1 - beta1**grad_t) )/(param_v_grad[i].sqrt() + epsilon)
        
    for param in parameters:
        param.grad.data.zero_()


def train(to_valid=False):
    """
        Main training loop
    """
    batch_num = len(train_loader)
    print("Total Batch number {}".format(batch_num))
    correct_ctr, valid_ctr = 0, 0
    for i_batch, train_batch in enumerate(train_loader):
        x = Variable(train_batch['x']).type(dtype).cuda()
        target = Variable(train_batch['y'].type(dtype)).cuda()
        

        if i_batch > batch_num * 0.9:
            # Validating
            if not to_valid:
                continue
            prob_y = model.out(x,is_test=True)
            loss = criterion(target, prob_y)
            correct_ctr += count_correct(train_batch['y'], prob_y)
            valid_ctr += train_batch['y'].size()[0]
        else:
            # Training
            prob_y = model.out(x,is_test=False)
            loss = criterion(target,prob_y)
            if(i_batch % 400 == 0):
                print("batch #{}, loss:{}".format(i_batch, loss.data.cpu().numpy()[0]))
            loss.backward()
            grad_update(parameters)
    if to_valid:
        print("Correct Count: {} ({}%),".format( correct_ctr, correct_ctr/valid_ctr*100))

def post_training():
    #print("Start Validation")
    #validation()
    print("Start Predicting...")
    print("Calculating {} elements".format(len(Dtest)))
    test(Dtest)

print("Loading Datasets...")
Dtrain = FbankDataset()
train_ds, valid_ds = validation_split(Dtrain,val_share=0.12)
train_loader = DataLoader(Dtrain, batch_size=nBatch, num_workers=6,pin_memory=True)
valid_loader = DataLoader(valid_ds, batch_size=len(valid_ds), num_workers=6, pin_memory=True)
Dtest = FbankDataset(is_test=True)
test_loader = DataLoader(Dtest, batch_size=nBatch)

D_in = Dtrain[0]['x'].shape[0]
D_out = Dtrain[0]['y'].shape[0]

print("The dataset dimension is ({} -> {})".format(D_in,D_out))

model = Network()
# initialization of network parameters
hidden_layer_num = [D_in, 1024,1024,1024,1024, D_out]
for i in range(len(hidden_layer_num)-1):
    in_dim = hidden_layer_num[i]
    out_dim = hidden_layer_num[i+1]
    W = Variable( (0.2*(0.5-torch.rand(in_dim, out_dim))).type(dtype).cuda(), requires_grad=True)
    b = Variable( (0.2*(0.5-torch.rand(1, out_dim))).type(dtype).cuda(), requires_grad=True)
    if i < len(hidden_layer_num)-2:
        ll = Layer.FCLayer(W,b,act_fnc=F.relu)
    else:
        ll = Layer.PredLayer(W,b)
    model.append(ll)



parameters = model.parameters()

param_m_grad = [
   torch.zeros(parm.data.size()).type(dtype).cuda() for parm in parameters
] # 1st moment vector

param_v_grad = [
   torch.zeros(parm.data.size()).type(dtype).cuda() for parm in parameters
] # 2nd moment vector

print("Start Training...")

try:
    for epoch in range(200):
        print("epoch #{}".format(epoch))
        train(to_valid=(epoch % 3 == 0))

except KeyboardInterrupt:
    post_training()
    exit()

post_training()