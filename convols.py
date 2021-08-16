


from torch import Tensor
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
    
    
class LightGConv(MessagePassing): 
    def __init__(self, in_channels, out_channels):
        super(LightGConv, self).__init__(aggr='mean')  

    def forward(self, x, edge_index):
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index, x=x)
        
    def message(self, x_j): 
        return  x_j
    
    def update(self, inputs: Tensor) -> Tensor: # Gamma
        return inputs




class LRGCCF(MessagePassing):
    def __init__(self, in_channels,out_channels):
        super(LRGCCF,self).__init__(aggr='mean')
        self.lin = torch.nn.Linear(in_channels, out_channels)
    def forward(self,x,edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes = x.size(0));
        
        return self.lin(self.propagate(edge_index,x=x))
    def message(self,x_j):
        return x_j
    def update(self,inputs: Tensor) -> Tensor:
        return inputs