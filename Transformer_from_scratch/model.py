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
    

class FeedforwardBlock(nn.Module):
    def __init__(self, d_mdoel:int ,d_ff:int ,dropout:float):
        super().__init__()    
        self.liner_1 = nn.Linear(d_mdoel ,d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.liner_2 = nn. Linear(d_ff , d_mdoel) # w2 and b2 


    def forward (self, x ):
        # (batch , seq_len , d_model) --> (batch ,seq_len , d_ff) -- > (bathc , seq_len , d_model)
        #return self.liner_2(self.dropout(torch.relu(self.liner_1(x))))

        h1 = self.liner_1(x)
        h2 = torch.relu(h1)
        h3 = self.dropout(h2)
        h4 = self.liner_2(h3)
        return h4 


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model :int , h:int , dropout:float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0 ,"d_model is not divisible by h "

        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model)  # Q
        self.w_k = nn.Linear(d_model, d_model)  # K
        self.w_v = nn.Linear(d_model, d_model)  # V

        self.w_o = nn.Linear(d_model ,d_model) # W_o
        self.dropout = nn.Dropout(dropout)

    def forward(self , q, k, v, mask): 
        query = self.w_q(q) #(Batch, seq_len, d_model) --> (Batch, seq_Len, d_model)
        key  = self.w_k(k) #(Batch, seq_len, d_model) --> (Batch, seq_Len, d_model)
        value = self.w_k(v) #(Batch, seq_len, d_model) --> (Batch, seq_Len, d_model)

        #(Bathc ,seq_len,d_model) --> (Batch,seq_len,h,d_k) -->(Batch,h, seq_len,d_k)
        query =query.view(query.shape[0] ,query.shape[1], self.h ,self.d_k).transpose(1,2)                   
        key = key.view(key.shape[0], key.shape[1] ,self.h , self.d_k).transpose(1,2)
        value = value.view(value.shape[0] ,value.shape[1] ,self.h ,self.d_k).transpose(1,2)

                                              
