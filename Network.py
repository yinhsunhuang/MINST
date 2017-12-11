class Network():
    def __init__(self):
        self.layers=[]
    def append(self, l):
        self.layers.append(l)
    def out(self,input, is_test=False):
        prev = input
        for l in self.layers:
            prev = l.out(prev, is_test)
        return prev
    def parameters(self):
        params = []
        for l in self.layers:
            for p in l.parameters():
                params.append(p)
        return params