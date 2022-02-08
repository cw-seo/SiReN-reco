
import time
import numpy as np
from torch.utils.data import Dataset
import torch
from tqdm import tqdm



class bipartite_dataset(Dataset):  
    def __init__(self, train,neg_dist,offset,num_u,num_v,K): 
        self.edge_1 = torch.tensor(train['userId'].values-1)
        self.edge_2 = torch.tensor(train['movieId'].values-1) +num_u
        self.edge_3 = torch.tensor(train['rating'].values) - offset
        self.neg_dist = neg_dist
        self.K = K;
        self.num_u = num_u
        self.num_v = num_v
        self.tot = np.arange(num_v)
        self.train = train
    def negs_gen_(self):
        print('negative sampling...'); st=time.time()
        self.edge_4 = torch.empty((len(self.edge_1),self.K),dtype=torch.long)
        prog = tqdm(desc='negative sampling for each epoch...',total=len(set(self.train['userId'].values)),position=0)
        for j in set(self.train['userId'].values):
            pos=self.train[self.train['userId']==j]['movieId'].values-1
            neg = np.setdiff1d(self.tot,pos)
            temp = (torch.tensor(np.random.choice(neg,len(pos)*self.K,replace=True,p=self.neg_dist[neg]/self.neg_dist[neg].sum()))+self.num_u).long()
            self.edge_4[self.edge_1==j-1]=temp.view(int(len(temp)/self.K),self.K)
            prog.update(1)
        prog.close()
        self.edge_4 = torch.tensor(self.edge_4).long()
        print('comlete ! %s'%(time.time()-st))
        
    def negs_gen_EP(self,epoch):
        print('negative sampling for next epochs...'); st=time.time()
        self.edge_4_tot = torch.empty((len(self.edge_1),self.K,epoch),dtype=torch.long)
        prog = tqdm(desc='negative sampling for next epochs...',total=len(set(self.train['userId'].values)),position=0)
        for j in set(self.train['userId'].values):
            pos=self.train[self.train['userId']==j]['movieId'].values-1
            neg = np.setdiff1d(self.tot,pos)
            temp = (torch.tensor(np.random.choice(neg,len(pos)*self.K*epoch,replace=True,p=self.neg_dist[neg]/self.neg_dist[neg].sum()))+self.num_u).long()
            self.edge_4_tot[self.edge_1==j-1]=temp.view(int(len(temp)/self.K/epoch),self.K,epoch)
            prog.update(1)
        prog.close()
        self.edge_4_tot = torch.tensor(self.edge_4_tot).long()
        print('comlete ! %s'%(time.time()-st))
    def __len__(self):
        return len(self.edge_1)
    def __getitem__(self,idx):
        u = self.edge_1[idx]
        v = self.edge_2[idx]
        w = self.edge_3[idx]
        negs = self.edge_4[idx]
        return u,v,w,negs


def deg_dist(train, num_v):
    uni, cou = np.unique(train['movieId'].values-1,return_counts=True)
    cou = cou**(0.75)
    deg = np.zeros(num_v)
    deg[uni] = cou
    return torch.tensor(deg)


def gen_top_k(data_class, r_hat, K=300):
    all_items = set(np.arange(1,data_class.num_v + 1))
    tot_items = set(data_class.train['movieId']).union(set(data_class.test['movieId']))
    no_items = all_items - tot_items;
    tot_items = torch.tensor(list(tot_items)) - 1
    no_items = (torch.tensor(list(no_items)) - 1).long()
    r_hat[:,no_items] = -np.inf
    for u,i in data_class.train.values[:,:-1]-1:
        r_hat[u,i] = -np.inf

    _, reco = torch.topk(r_hat,K);
    reco = reco.numpy()
    
    return reco