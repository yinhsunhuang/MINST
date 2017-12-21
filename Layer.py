import torch
import torch.nn.functional as F
from torch.autograd import Variable

class Layer():
    def __init__(self):
        self.parameter = []
    def out(self):
        pass

class FCLayer(Layer):
    def __init__(self, W, b, act_fnc=F.sigmoid):
        self.W = W
        self.b = b
        self.act_fnc = act_fnc
        self.params = [self.W, self.b]

    def out(self, input, is_test):
        if self.act_fnc != None:
            return self.act_fnc(input.mm(self.W)+ self.b)
        else:
            return input.mm(self.W) + self.b
    
    def parameters(self):
        return self.params
    def __str__(self):
        return "FC Linear Layer {} -> {}".format(self.W.data.shape[0],self.W.data.shape[1])

class DropoutLayer(Layer):
    def __init__(self, dropout=0.5):
        self.dropout=dropout

    def out(self, input, is_test):
        if not is_test:
            mask = Variable(torch.bernoulli(torch.ones(1,input.size()[1])*(1 - self.dropout)).cuda())
            return input * mask
        else:
            return input * (1 - self.dropout)
    
    def parameters(self):
        return []
    def __str__(self):
        return "Dropout Layer w.p. {}".format(self.dropout)

class PredLayer(FCLayer):
    def __init__(self, W, b, act_fnc=F.log_softmax):
        super().__init__(W,b,act_fnc)
    def __str__(self):
        return "Prediction Layer(Log-softmax) {} -> {}".format(self.W.data.shape[0], self.W.data.shape[1])
