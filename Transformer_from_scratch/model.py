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

    @staticmethod
    def attention(query , key , value , mask , dropout: nn.Dropout):
        d_k = query.shape[-1]

        # (Batch , h, seq_len, d_k) -- > (Batch , seq_len , seq_len)
        attention_score = query @ key.transpose(-2 , -1) / math.sqrt(d_k)  #)(-2 , -1) this chage the (seq_Len, d_model)  --> (d_model,seq_Len)
        if mask is not None:
            attention_score.masked_fill_(mask == 0, -1e9) # in that ware this (mask == 0 true the replace with this value -1e9
        attention_score = attention_score.softmax(dim = -1)    #(Batch , h , seq_len,seq_len)
        if dropout is not None:
            attention_score = dropout(attention_score)

        return (attention_score @ value) ,attention_score   # that second used for ther visualizing and dubug
     



    def forward(self , q, k, v, mask): 
        query = self.w_q(q) #(Batch, seq_len, d_model) --> (Batch, seq_Len, d_model)
        key  = self.w_k(k) #(Batch, seq_len, d_model) --> (Batch, seq_Len, d_model)
        value = self.w_v(v) #(Batch, seq_len, d_model) --> (Batch, seq_Len, d_model)

        #(Bathc ,seq_len,d_model) --> (Batch,seq_len,h,d_k) -->(Batch,h, seq_len,d_k)
        query =query.view(query.shape[0] ,query.shape[1], self.h ,self.d_k).transpose(1,2)                   
        key = key.view(key.shape[0], key.shape[1] ,self.h , self.d_k).transpose(1,2)
        value = value.view(value.shape[0] ,value.shape[1] ,self.h ,self.d_k).transpose(1,2)

        x , self.attention_score = MultiHeadAttentionBlock.attention(query , key , value , mask , self.dropout)                                  

        #(Batch , h , seq_len ,d_k) --> (Batch ,seq_len , d_model) --> (Batch ,seq_len , d_model)

        x =x.transpose(1,2).contiguous().view(x.shape[0] , -1 ,self.h * self.d_k)

        # (Batch , seq_len , d_model) --> (Batch,seq_len ,d_model)
        return self.w_o(x)                      
    

class ResidualConnection(nn.Module):
    def __init__(self, dropout :float):
        super().__init__()
        self.dropout =nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward (self , x ,sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
    
class EncoderBlock(nn.module):
    def __init__(self,self_attention_block :MultiHeadAttentionBlock , feed_forward_black:FeedforwardBlock , dropout: float):
        super().__init__()
        self.self_attention_block = MultiHeadAttentionBlock
        self.feed_forward_block = FeedforwardBlock
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])


    def forward(self , x ,src_mask ) :
        x =   self.residual_connections[0](x , lambda x: self.self_attention_block(x ,x , x ,src_mask)) 
        x = self.residual_connections[1] (x ,self.feed_forward_block)
        return x
    
class Encoder(nn.Module):
    def __init__(self, layer:nn.ModuleList):
        super().__init__() 
        self.layer = layer 
        self.norm = LayerNormalization()

    def forward(self , x ,mask ):
        for layer in self.layer:
            x = layer(x,mask )
        return self.norm(x)    
        
class decoderBlock(nn.Module):
    def __init__(self,self_attentation_block :MultiHeadAttentionBlock , cross_attention:MultiHeadAttentionBlock , feed_forward_block :FeedforwardBlock, dropout:float):
        super().__init__()    
        self.self_attentaion_block = self_attentation_block
        self.cross_atteation_block = cross_attention
        self.feed_forward_bloack = feed_forward_block
        self.residual_connections = nn.Module([ResidualConnection(dropout )for  _ in range(3)])

    def forward ( self, x ,encoder_output ,src_mask ,tgt_mask ):
            x = self.residual_connections[0](x ,lambda x: self.self_attentaion_block(x , x , x ,tgt_mask))
            x = self.residual_connections[1] (x ,lambda x: self.cross_atteation_block(x , encoder_output ,encoder_output , src_mask))
            x = self.residual_connections[2] (x , self.feed_forward_bloack)
            return x
    
class decoder (nn.Module):
    def __init__(self, layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward (self ,x ,encoder_output , src_mask , tgt_mask  ):
        for layer in self.layers:
            x = layer(x , encoder_output ,src_mask,tgt_mask)
        return self.norm(x) 
            
