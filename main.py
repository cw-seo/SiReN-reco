import time
import torch
from torch.utils.data import DataLoader
from torch import optim

from tqdm import tqdm
from evaluator import evaluate as eva 
from evaluator import evaluator as ev
from util import  bipartite_dataset, deg_dist,gen_top_k
from data_loader import Data_loader
from siren import SiReN
import argparse

def main(args):
    data_class=Data_loader(args.dataset,args.version)
    threshold = round(args.offset) # To generate ground truth set 
    print('data loading...');st=time.time()
    train,test = data_class.data_load();
    train = train.astype({'userId':'int64', 'movieId':'int64'})
    data_class.train = train; data_class.test = test
    print('loading complete! time :: %s'%(time.time()-st))
    
    
    print('generate negative candidates...'); st=time.time()
    neg_dist = deg_dist(train,data_class.num_v)
    print('complete ! time : %s'%(time.time()-st))    
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model= SiReN(train, data_class.num_u,data_class.num_v,offset=args.offset,num_layers = args.num_layers,MLP_layers=args.MLP_layers,dim=args.dim,device=device,reg=args.reg)
    model.data_p.to(device)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    
    
    print("\nTraining on {}...\n".format(device))
    model.train()
    training_dataset=bipartite_dataset(train,neg_dist,args.offset,data_class.num_u,data_class.num_v,args.K);
    
    for EPOCH in range(1,args.epoch+1):
        if EPOCH%20-1==0:training_dataset.negs_gen_EP(20)
            
        
        LOSS=0
        training_dataset.edge_4 = training_dataset.edge_4_tot[:,:,EPOCH%20-1]
        
        ds = DataLoader(training_dataset,batch_size=args.batch_size,shuffle=True)
        q=0
        pbar = tqdm(desc = 'Version : {} Epoch {}/{}'.format(args.version,EPOCH,args.epoch),total=len(ds),position=0)
        
        for u,v,w,negs in ds:   
            q+=len(u)
            st=time.time()
            optimizer.zero_grad()
            loss = model(u,v,w,negs,device) # original
            loss.backward()                
            optimizer.step()
            LOSS+=loss.item() * len(ds)
            
            pbar.update(1);
            pbar.set_postfix({'loss':loss.item()})

        pbar.close()

        if EPOCH%20 ==0:

            model.eval()
            emb = model.aggregate();
            emb_u, emb_v = torch.split(emb,[data_class.num_u,data_class.num_v])
            emb_u = emb_u.cpu().detach(); emb_v = emb_v.cpu().detach()
            r_hat = emb_u.mm(emb_v.t())
            reco = gen_top_k(data_class,r_hat)
            eval_ = ev(data_class,reco,args)
            eval_.precision_and_recall()
            eval_.normalized_DCG()
            print("\n***************************************************************************************")
            print(" /* Recommendation Accuracy */")
            print('N :: %s'%(eval_.N))
            print('Precision at :: %s'%(eval_.N),eval_.p['total'][eval_.N-1])
            print('Recall at [10, 15, 20] :: ',eval_.r['total'][eval_.N-1])
            print('nDCG at [10, 15, 20] :: ',eval_.nDCG['total'][eval_.N-1])
            print("***************************************************************************************")
            model.train()



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        type = str,
                        default = 'ML-1M',
                        help = "Dataset"
                        )
    parser.add_argument('--version',
                        type = int,
                        default =1,
                        help = "Dataset version"
                        )
    parser.add_argument('--batch_size',
                        type = int,
                        default = 1024,
                        help = "Batch size"
                        )

    parser.add_argument('--dim',
                        type = int,
                        default = 64,
                        help = "Dimension"
                        )
    parser.add_argument('--lr',
                        type = float,
                        default = 5e-3,
                        help = "Learning rate"
                        )
    parser.add_argument('--offset',
                        type = float,
                        default = 3.5,
                        help = "Criterion of likes/dislikes"
                        )
    parser.add_argument('--K',
                        type = int,
                        default = 40,
                        help = "The number of negative samples"
                        )
    parser.add_argument('--num_layers',
                        type = int,
                        default = 4,
                        help = "The number of layers of a GNN model for the graph with positive edges"
                        )
    parser.add_argument('--MLP_layers',
                        type = int,
                        default = 2,
                        help = "The number of layers of MLP for the graph with negative edges"
                        )
    parser.add_argument('--epoch',
                        type = int,
                        default = 1000,
                        help = "The number of epochs"
                        )
    parser.add_argument('--reg',
                        type = float,
                        default = 0.05,
                        help = "Regularization coefficient"
                        )
    args = parser.parse_args()
    main(args)