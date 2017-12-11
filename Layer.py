import torch
import torch.nn.functional as F
from torch.autograd import Variable

class Layer():
    def __init__(self):
        self.parameter = []
    def out(self):
        pass

class FCLayer(Layer):
    def __init__(self, W, b, act_fnc=F.sigmoid, dropout=0.5):
        self.W = W
        self.b = b
        self.act_fnc = act_fnc
        self.dropout = dropout
        self.params = [self.W, self.b]

    def out(self, input, is_test):
        z = input.mm(self.W)
        if not is_test:
            mask = Variable(torch.bernoulli(torch.ones(z.size())*self.dropout).cuda())
            return self.act_fnc(z * mask + self.b)
        else:
            return self.act_fnc((z+ self.b)*self.dropout)
    
    def parameters(self):
        return self.params

class PredLayer(Layer):
    def __init__(self, W, b, act_fnc=F.log_softmax):
        self.W = W
        self.b = b
        self.act_fnc = act_fnc
        self.params = [self.W, self.b]

    def out(self, input, is_test):
        return self.act_fnc(input.mm(self.W) + self.b)
    
    def parameters(self):
        return self.params