# SiReN



## This is our PyTorch implementation code for our paper:





Note. In this implementation, we use LightGCN (X. He et al., SIGIR'20) as the GNN model for the graph with positive edges in our SiReN method. 





## Environment Requirements

The code has been tested under Python 3.7.7. The required packages are as follows:

* Pytorch == 1.5.0
* Pytorch Geometric == 1.6.3





## Example : ML-1M dataset

```python
python main.py --dataset ML-1M --version 1 --reg 0.1
```





## Example : Amazon-Book dataset

```python
python main.py --dataset amazon --version 1 --reg 0.05
```





## Example : Yelp dataset

```python
python main.py --dataset yelp --version 1 --reg 0.05
```

