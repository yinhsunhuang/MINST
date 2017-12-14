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
        return self.act_fnc(input.mm(self.W)+ self.b)
    
    def parameters(self):
        return self.params

class DropoutLayer(Layer):
    def __init__(self, dropout=0.5):
        self.dropout=1-dropout

    def out(self, input, is_test):
        if not is_test:
            mask = Variable(torch.bernoulli(torch.ones(1,input.size()[1])*self.dropout).cuda())
            return input * mask
        else:
            return input * self.dropout
    
    def parameters(self):
        return []

class PredLayer(FCLayer):
    def __init__(self, W, b, act_fnc=F.log_softmax):
        super().__init__(W,b,act_fnc)