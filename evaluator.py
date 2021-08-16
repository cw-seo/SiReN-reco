
import numpy as np
class evaluate():
    def __init__(self,reco,train,test,threshold,num_u,num_v,N=[5,10,15,20,25],ratings=[20,50]):
        '''
        train : training set
        test : test set
        threshold : To generate ground truth set from test set
        '''

        self.reco = reco
        self.num_u = num_u;
        self.num_v = num_v;
        self.N=N
        self.p=[]
        self.r=[]
        self.NDCG=[]
        self.p_c1=[]; self.p_c2=[]; self.p_c3=[]
        self.r_c1=[]; self.r_c2=[]; self.r_c3=[]
        self.NDCG_c1=[]; self.NDCG_c2=[]; self.NDCG_c3=[]
        
        
        self.tr = train; self.te = test;
        
        self.threshold = threshold;
        self.gen_ground_truth_set()
        self.ratings = ratings
        self.partition_into_groups_(self.ratings)
        print('\nevaluating recommendation accuracy....')
        self.precision_and_recall_G(self.group1,1)
        self.precision_and_recall_G(self.group2,2)
        self.precision_and_recall_G(self.group3,3)

        self.Normalized_DCG_G(self.group1,1)
        self.Normalized_DCG_G(self.group2,2)
        self.Normalized_DCG_G(self.group3,3)
        self.metric_total()

    def gen_ground_truth_set(self):
        result = dict()
        GT = self.te[self.te['rating']>=self.threshold];
        U = set(GT['userId'])
        for i in U:
            result[i] = list(set([j for j in GT[GT['userId']==i]['movieId']]))#-set(self.TOP))
            if len(result[i])==0:
                del(result[i])
        self.GT = result

    def precision_and_recall(self):
        user_in_GT=[j for j in self.GT];
        for n in self.N:
            p=0; r=0;
            for i in user_in_GT:
                topn=self.reco[i][:n]
                num_hit=len(set(topn).intersection(set(self.GT[i])));
                p+=num_hit/n; r+=num_hit/len(self.GT[i]);
            self.p.append(p/len(user_in_GT)); self.r.append(r/len(user_in_GT));
                
    def Normalized_DCG(self):
        maxn=max(self.N);
        user_in_GT=[j for j in self.GT];
        ndcg=np.zeros(maxn);
        
        for i in user_in_GT:
            idcg_len = min(len(self.GT[i]), maxn)
            temp_idcg = np.cumsum(1.0 / np.log2(np.arange(2, maxn + 2)))
            temp_idcg[idcg_len:] = temp_idcg[idcg_len-1]
            temp_dcg=np.cumsum([1.0/np.log2(idx+2) if item in self.GT[i] else 0.0 for idx, item in enumerate(self.reco[i][:maxn])])
            ndcg+=temp_dcg/temp_idcg;
        ndcg/=len(user_in_GT);
        for n in self.N:
            self.NDCG.append(ndcg[n-1])
            
            
    def metric_total(self):
        self.p = self.len1 * np.array(self.p_c1) + self.len2 * np.array(self.p_c2) + self.len3 * np.array(self.p_c3);
        self.p/= self.len1 + self.len2 + self.len3
        self.p = list(self.p)
        
        self.r = self.len1 * np.array(self.r_c1) + self.len2 * np.array(self.r_c2) + self.len3 * np.array(self.r_c3);
        self.r/= self.len1 + self.len2 + self.len3
        self.r = list(self.r)
        
        self.NDCG = self.len1 * np.array(self.NDCG_c1) + self.len2 * np.array(self.NDCG_c2) + self.len3 * np.array(self.NDCG_c3);
        self.NDCG/= self.len1 + self.len2 + self.len3
        self.NDCG = list(self.NDCG)

    def partition_into_groups_(self,ratings=[20,50]):
        unique_u, counts_u = np.unique(self.tr['userId'].values,return_counts=True)
        self.group1 = unique_u[np.argwhere(counts_u<ratings[0])]
        temp = unique_u[np.argwhere(counts_u<ratings[1])]
        self.group2 = np.setdiff1d(temp,self.group1)
        self.group3 = np.setdiff1d(unique_u,temp)
        self.cold_groups = ratings
        self.group1 = list(self.group1.reshape(-1))
        self.group2 = list(self.group2.reshape(-1))
        self.group3 = list(self.group3.reshape(-1))
    
    def precision_and_recall_G(self,group,gn):
        user_in_GT=[j for j in self.GT];
        leng = 0 ; maxn = max(self.N) ; p = np.zeros(maxn); r = np.zeros(maxn);
        for i in user_in_GT:
            if i in group:
                
                leng+=1
                hit_ = np.cumsum([1.0 if item in self.GT[i] else 0.0 for idx, item in enumerate(self.reco[i][:maxn])])
                p+=hit_ / np.arange(1,maxn+1); r+=hit_/len(self.GT[i])
        p/= leng; r/=leng;
        for n in self.N:
            if gn == 1 :
                self.p_c1.append(p[n-1])
                self.r_c1.append(r[n-1])
                self.len1 = leng;
            elif gn == 2 :
                self.p_c2.append(p[n-1])
                self.r_c2.append(r[n-1])
                self.len2 = leng;
            elif gn == 3 :
                self.p_c3.append(p[n-1])
                self.r_c3.append(r[n-1])
                self.len3 = leng;
            
    def Normalized_DCG_G(self,group,gn):
        maxn=max(self.N);
        user_in_GT=[j for j in self.GT];
        ndcg=np.zeros(maxn);
        leng = 0
        for i in user_in_GT:
            if i in group:
                leng+=1
                idcg_len = min(len(self.GT[i]), maxn)
                temp_idcg = np.cumsum(1.0 / np.log2(np.arange(2, maxn + 2)))
                temp_idcg[idcg_len:] = temp_idcg[idcg_len-1]
                temp_dcg=np.cumsum([1.0/np.log2(idx+2) if item in self.GT[i] else 0.0 for idx, item in enumerate(self.reco[i][:maxn])])
                ndcg+=temp_dcg/temp_idcg;
        ndcg/=leng
        for n in self.N:
            if gn == 1 :
                self.NDCG_c1.append(ndcg[n-1])
            elif gn == 2 :
                self.NDCG_c2.append(ndcg[n-1])
            elif gn == 3 :
                self.NDCG_c3.append(ndcg[n-1])