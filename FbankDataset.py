import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
class FbankDataset(Dataset):
    """FbankD Dataset"""
    def __init__(self, base_dir='MLDS_HW1_RELEASE_v1/', is_test=False):
        self._loaded = False
        self._base_dir = base_dir
        self._is_test = is_test
        self.loadMap()
        self.loadData()
        super(FbankDataset, self).__init__()

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        if not self._is_test:
            sample = {
                'label': self.frame_names[idx],
                'x': self.x[idx:idx+9,:].flatten(),
                'y': self.y[idx]
            }
        else:
            sample = {
                'label': self.frame_names[idx],
                'x':self.x[idx:idx+9,:].flatten()
            }
        return sample

    def phone_to_onehot(self,x):
        id = self.phone_ctr[x[0]]
        out = np.zeros(48,np.int8)
        out[id]=1
        return out

    def state_to_onehot(self,x):
        out = np.zeros(1943,np.int8)
        out[x]=1
        return out
        
    def loadData(self,ratio=0.1):
        '''Helper function for loading training data and labels'''
        if not self._is_test:
            self.fbank_train = pd.read_csv(self._base_dir + 'fbank/train.ark', index_col=0, sep=' ', header=None)
        else:
            self.fbank_train = pd.read_csv(self._base_dir + 'fbank/test.ark', index_col=0, sep=' ', header=None)
            
        self.frame_names = self.fbank_train.index.tolist()
        self.x = self.fbank_train.values
        self.N = self.x.shape[0]

        self.x = np.vstack( ( np.zeros((4,self.x.shape[1])), self.x ,np.zeros((4,self.x.shape[1])) ) )
        
        if not self._is_test:
            self.labels = pd.read_csv(self._base_dir+'state_label/train.lab',index_col=0, sep=',', header=None)
            self.y = self.labels.ix[self.frame_names, :].values
            self.y = np.asarray([self.state_to_onehot(yi) for yi in self.y],dtype=np.uint8)


    def loadMap(self):
        # 48-39 map
        self.phone_map={}
        self.phone_ctr={}
        self.ctr_phone={}
        ctr=0
        with open(self._base_dir+'phones/48_39.map', 'r') as f:
            for l in f:
                ll = l.rstrip().split('\t')
                self.phone_map[ll[0]] = ll[1]
                self.phone_ctr[ll[0]] = ctr
                self.ctr_phone[ctr]=ll[1]
                ctr = ctr + 1

        self.state_39={}
        with open(self._base_dir+'phones/state_48_39.map', 'r') as f:
            for l in f:
                ll = l.rstrip().split('\t')
                self.state_39[int(ll[0])] = ll[2]

class PartialDataset(Dataset):
    def __init__(self, parent_ds, offset, length):
        self.parent_ds = parent_ds
        self.offset = offset
        self.length = length
        self.perm = torch.randperm(len(self.parent_ds))
        assert len(parent_ds)>=offset+length, Exception("Parent Dataset not long enough")
        super(PartialDataset, self).__init__()
    def __len__(self):
        return self.length
    def __getitem__(self, i):
        return self.parent_ds[self.perm[i+self.offset]]

def validation_split(dataset, val_share=0.1):
    val_offset = int(len(dataset)*(1-val_share))
    return PartialDataset(dataset, 0, val_offset), PartialDataset(dataset, val_offset, len(dataset)-val_offset)

if __name__=='__main__':
    Dtest = FbankDataset(is_test=False)
    print("Testing containing {} test cases".format(len(Dtest)))
    Dtrain = FbankDataset()
    for i in range(len(Dtrain)):
        print(Dtrain[i]['x'])
        print(Dtrain[i]['y'])
        print(Dtrain[i]['label'])
    train_ds, valid_ds = validation_split(Dtrain)
    train_loader = DataLoader(train_ds, batch_size=4)
    valid_loader = DataLoader(valid_ds, batch_size=4)
    print()
    for i_batch, sample_batched in enumerate(train_loader):
        print(i_batch, sample_batched['label'])
    print("jasdlkfjalksdjflkasdjlkfj")
    for i_batch, sample_batched in enumerate(train_loader):
        print(i_batch, sample_batched['label'])

    for i_batch, sample_batched in enumerate(valid_loader):
        print(i_batch, sample_batched['label'])
    print("jasdlkfjalksdjflkasdjlkfj")
    for i_batch, sample_batched in enumerate(valid_loader):
        print(i_batch, sample_batched['label'])