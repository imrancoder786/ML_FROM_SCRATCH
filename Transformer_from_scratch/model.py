import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    #d_model = size of the vector .

    def __init__(self, d_model:int ,vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size , d_model)

    def forword(self, x): #(embedding)vector of size 512
        return self.embedding(x) * math.sqrt(self.d_model)
    

""" 
The positional encoding same vector size 512 only conver once and only
computeed once and resue for the every sentence during 
tranning and inference
""" 
class PositionalEncoding(nn.module):

    def __inti__(self, d_model:int , seq_length:int ,dropout:float):
        super().__init__()
        self.d_model = d_model
        self.seq_lenght = seq_length
        self.dropout = dropout

        #create a matrix of shape (seq_len , d_model)
        pe = torch.zeros(seq_length , d_model)
        # create a vector of shape (seq_len ,1)
        position = torch.arange(0 , seq_length , dtype=torch.float).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0 , d_model , 2).float() * (-math.log(10000.0) / d_model ))

        #apply the sin to even position
        pe[: , 0::2]  = torch.sin(position  * div_term)
        pe[: , 1::2]  = torch.cos(position * div_term)

        pe = pe.squeeze(0) # that become the (1,seq_lenth , d_model)

        self.register_buffer('pe' , pe)  # that used if we want tge model to save but not as a parameter.

    def forward(self , x):
        x = x + (self.pe[:, :x.shape[1] , :]).requires_grad_(False)  # so that model not learning this for handling the srq_;enght and inductive bias , to produce ne w and diffreent lenght of the word 
        return self.dropout(x)
    
class LayerNormalization(nn.module):
    def __init__(self , eps: float = 10 **-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.parameter(torch.ones(1)) # multiplied
        self.bias = nn.parameter(torch.zeros(1))  #added

    def forward(self ,x):
        mean = x.mean(dim = -1 ,keepdim= True)   
        std = x.std(dim = -1 ,keepdim=True)

        return self.alpha * ( x - mean ) / (std + self.eps) +self.bias