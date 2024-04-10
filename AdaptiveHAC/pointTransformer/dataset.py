import numpy as np
import os
from torch.utils.data import Dataset
import torch


class ModelNetDataLoader(Dataset):
    def __init__(self, root, npoint=1024, split='train', cache_size=15000):
        self.root = root
        self.npoints = npoint
        self.catfile = os.path.join(self.root, 'class_names.txt')
        self.cat = [line.rstrip() for line in open(self.catfile)]   # class names
        print(self.cat) #['boxing', 'jump', 'jack', 'squats', 'walk']

        self.classes = dict(zip(self.cat, range(len(self.cat)))) # dict(zip(a,b)) a=key b=value
        print(self.classes) #  {'boxing': 0, 'jump': 1, 'jack': 2, 'squats': 3, 'walk': 4}
    
        shape_ids = {}  
        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'train_set.txt'))] # file names for training
        #print(shape_ids['train'])
        shape_ids['test'] =  [line.rstrip() for line in open(os.path.join(self.root, 'test_set.txt'))]
        #print(shape_ids['test'])

        assert (split == 'train' or split == 'test')
        labels = [x.split('/')[-1].split('_')[0] for x in shape_ids[split]] # keep the label(class name)  label is the name of a file
        
        # list of (label, shape_txt_file_path) tuple  (label, file path)
        print(len(shape_ids[split]))
        self.datapath = [(labels[i], os.path.join(self.root, shape_ids[split][i]).replace("\\","/") + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d'%(split,len(self.datapath)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            fn = self.datapath[index] # (label, path)
            cls = self.classes[self.datapath[index][0]] # 0 - 39
            cls = np.array([cls]).astype(np.int32) 
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32) # type : ndarry
            point_set = point_set[0:self.npoints,:]

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)

        return point_set, cls

    def __getitem__(self, index):
        return self._get_item(index)

class PCModelNetDataLoader(Dataset):
    def __init__(self, PC, npoint=1024, cache_size=15000):
        self.classes = dict(zip(PC[0][0].activities, range(len(PC[0][0].activities))))
        self.npoint = npoint
        self.cache_size = cache_size
        self.PC = PC
        
    def __len__(self):
        return len(self.PC)
    
    def __getitem__(self, index):
        cls = self.PC[index][0].mean_label
        cls = torch.from_numpy(np.array([cls]).astype(np.int32)) 
        point_set = torch.from_numpy(self.PC[index][0].data[:,0:5].astype(np.float32)) # need workaround for [0]
        return point_set, cls
    
class SeqModelNetDataLoader(Dataset):
    def __init__(self, root):
        self.root = root
        self.catfile = os.path.join(self.root, 'class_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]   # class names
        print(self.cat) #['boxing', 'jump', 'jack', 'squats', 'walk']

        self.classes = dict(zip(self.cat, range(len(self.cat)))) # dict(zip(a,b)) a=key b=value
        print(self.classes) #  {'boxing': 0, 'jump': 1, 'jack': 2, 'squats': 3, 'walk': 4}
    
        shape_ids = {}  
        shape_ids =  [line.rstrip() for line in open(os.path.join(self.root, 'test_set.txt'))]
        #print(shape_ids['test'])

        labels = [x.split('/')[-1].split('_')[0] for x in shape_ids] # keep the label(class name)  label is the name of a file
        
        # list of (label, shape_txt_file_path) tuple  (label, file path)
        print(len(shape_ids))
        self.datapath = [(labels[i], os.path.join(self.root, shape_ids[i]).replace("\\","/") + '.txt') for i
                         in range(len(shape_ids))]
        print('The size of test data is %d'%(len(self.datapath)))

        self.nseq = max([int(x[1].split('/')[-3]) for x in self.datapath])

    def __len__(self):
        return len(self.datapath)
    
    def __getitem__(self, index):
            fn = self.datapath[index] # (label, path)
            cls = self.classes[self.datapath[index][0]] # 0 - 39
            node = int(self.datapath[index][1].split('/')[-2])
            sequence = int(self.datapath[index][1].split('/')[-3])
            cls = torch.from_numpy(np.array([cls]).astype(np.int32)) 
            point_set = torch.from_numpy(np.loadtxt(fn[1], delimiter=',').astype(np.float32)) # type : ndarry
            return point_set, cls, sequence, node


if __name__ == '__main__':
    data = SeqModelNetDataLoader('scripts/baseline/data/')
    print(len(data))
    print(data)
    # print(DataLoader)
    # for point,label in DataLoader:
    #     print(point.shape)
    #     print(label.shape)
    # tup , class_name = data._get_item(1025)
    # print(len(tup[0]))
    # print(data.classes)
    # print(len(data._get_item(1)))
