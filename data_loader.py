import pandas as pd

class Data_loader():
    def __init__(self,dataset,version):
        self.dataset=dataset; self.version=version
        if dataset=='ML-1M':
            self.sep='::'
            self.names=['userId','movieId','rating','timestemp'];
            
            self.path_for_whole='./ml-1m/ratings.dat'
            self.path_for_train='./ml-1m/train_1m%s.dat'%(version)
            self.path_for_test='./ml-1m/test_1m%s.dat'%(version)
            self.num_u=6040; self.num_v=3952;
            
        
        elif dataset=='amazon':
            self.path_for_whole='./amazon-book/amazon-books-enc.csv'
            self.path_for_train='./amazon-book/train_amazon%s.dat'%(version)
            self.path_for_test='./amazon-book/test_amazon%s.dat'%(version)
            self.num_u=35736; self.num_v=38121;

        elif dataset=='yelp':
            self.path_for_whole='./yelp/YELP_encoded.csv'
            self.path_for_train='./yelp/train_yelp%s.dat'%(version)
            self.path_for_test='./yelp/test_yelp%s.dat'%(version)
            self.num_u=41772; self.num_v=30037;

        else:
            raise NotImplementedError("incorrect dataset, you can choose the dataset in ('ML-100K','ML-1M','amazon','yelp')")
        
        

    def data_load(self):
        if self.dataset=='ML-1M':
            self.whole_=pd.read_csv(self.path_for_whole, names = self.names, sep=self.sep, engine='python').drop('timestemp',axis=1).sample(frac=1,replace=False,random_state=self.version)
            self.train_set = pd.read_csv(self.path_for_train,engine='python',names=self.names).drop('timestemp',axis=1)
            self.test_set = pd.read_csv(self.path_for_test,engine='python',names=self.names).drop('timestemp',axis=1)            
                
        elif self.dataset=='amazon':
            self.whole_=pd.read_csv(self.path_for_whole,index_col=0).sample(frac=1,replace=False);
            self.train_set=pd.read_csv(self.path_for_train,index_col=0)
            self.test_set=pd.read_csv(self.path_for_test,index_col=0)
            

        elif self.dataset=='yelp':
            self.whole_=pd.read_csv(self.path_for_whole,index_col=0).sample(frac=1,replace=False);
            self.train_set=pd.read_csv(self.path_for_train,index_col=0)
            self.test_set=pd.read_csv(self.path_for_test,index_col=0)
        
        return self.train_set, self.test_set
    
    