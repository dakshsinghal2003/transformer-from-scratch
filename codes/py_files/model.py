import torch 
from torch import nn




import torch
import torch.nn as nn
from typing import List,OrderedDict,Dict,Tuple,Literal
from math import sqrt,pow,sin,cos
class InputEmbeddings(nn.Module):
    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(num_embeddings=self.vocab_size,
                                       embedding_dim=self.vocab_size)
    def forward(self,x):
        return self.embeddings(x)*sqrt(self.d_model)
    # TODO : Find the reason for mutiplying with the sqrt(d_model)
        















class PositionalEncoding(nn.Module):
    def __init__(self,d_model:int,sequence_length:int,p_drop:float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.sequence_length = sequence_length
        self.dropout = nn.Dropout()
        positional_embeddings = torch.zeros(size=(self.sequence_length,self.d_model),dtype=torch.float64)
        for i in range(self.d_model):
            for j in range(self.sequence_length):
                if i%2==0:
                    omega = torch.tensor(pow(1000,i/self.d_model))
                    
                    positional_embeddings[j][i] = sin(j/omega)
                else:
                    omega = torch.tensor(pow(1000,i-1/self.d_model))
                    positional_embeddings[j][i] = torch.cos(j/omega)
        positional_embeddings = positional_embeddings.unsqueeze(0)
        self.register_buffer(name='positional_embeddings',tensor=positional_embeddings)
        
        
    def forward(self,sequence:torch.TensorType):
        if sequence.ndim==2:
            sequence =sequence.unsqueeze(0)
        assert sequence.shape[2] == self.d_model,f"""The embedding dimensions of model and input dont match model's dimensions : {self.d_model} , input dimensions : {sequence.shape[0]}"""
        sequence =  sequence + (self.positional_embeddings[:,:sequence.shape[1],:]).requires_grad_(False)
        return self.dropout(sequence)
pe = PositionalEncoding(d_model=4,sequence_length=3,p_drop=0.3) # 3 word sequence with each word represented as a vector of 4 numbers














class FeedForwardNeuralNetwork(nn.Module):
    def __init__(self,input_dim:int = 512, output_dim = 2048, p_dropout:float = 0.1):
        super().__init__()
        self.linear_layer1 = nn.Linear(in_features=input_dim,out_features=output_dim)
        self.linear_layer2 = nn.Linear(in_features=output_dim,out_features=input_dim)
        self.dropout = nn.Dropout(p = p_dropout)
    def forward(self,x:torch.TensorType):
        return self.linear_layer2(self.dropout(self.linear_layer1(x)))















class MultiHeadAttention(nn.Module):
    def __init__(self,d_model:int = 512,h:int = 8, dropout:float = 0.1):
        super().__init__()
        self.d_model = d_model  # expected input dimesions in each word
        
        self.h  = h   # number of heads 
        assert d_model%h == 0 ,"Expected values of d_model and h such that d_model is divisible by h"
        self.d_key = d_model//h
        self.w_query = nn.Linear(d_model,d_model) # weight matrix for query 
        self.w_key = nn.Linear(d_model,d_model) # weight matrix for key 
        
        
        # Note that here we're  going for d_model X d_model and not for d_model X (h*d_value)
        # Since we're given an input matrix of d_model.  
        self.w_value = nn.Linear(d_model,d_model) # weight matrix for value 
        self.w_o = nn.Linear(d_model,d_model)
    
    
    @staticmethod # this is a static method and can be accessed outside the class using this: MultiHeadAttention.attention(input parameters)
    def attention(self,key:torch.Tensor,query:torch.Tensor,value:torch.Tensor,mask):
        key_query = torch.matmul(query,key.T)
        scaled = key_query/torch.sqrt(self.d_key)  
        # At this point , the shape of the matrix is : (Batch , h , seqeuence_length , seqeuence_length)
        if mask is not None:
            scaled.masked_fill(mask==0,-1e9)
            
        softmax_ = scaled.softmax(dim=-1)
        
        return torch.matmul(softmax_,value)
    
    
    
    def forward(self,query,key,value,mask):
        q_w = self.w_query(query) # query matrix for input sequence , Shape : (Batch , sequence_length , d_model) 
        k_w = self.w_key(key)     # key matrix for input sequence   , Shape : (Batch , sequence_length , d_model)
        v_w = self.w_value(value) # value matrix for input sequence , Shape : (Batch , sequence_length , d_model)
        batch,sequence_length,d_model = q_w.shape
        
        # This is done to break the resultant query matrix of the sequence into h matrices
        # transpose is mainly done to make the matrix head-based indexable. 
        q_q_head_wise = q_w.view((batch,sequence_length,self.h,self.d_key).transpose(2,1)) 
        q_k_head_wise = k_w.view((batch,sequence_length,self.h,self.d_key).transpose(2,1)) 
        q_v_head_wise = v_w.view((batch,sequence_length,self.h,self.d_key).transpose(2,1)) 
        attention_score = MultiHeadAttention.attention(self,q_k_head_wise,q_q_head_wise,q_v_head_wise,mask)
        
        #TODO : Study contiguous memory allocation in detail. 
        
        attention_score = attention_score.transpose(1,2).contiguous().view(batch,sequence_length,self.h*self.d_key)
        # Tanspose to get the dimension normally indexable and merging the tensors back. 
        
        return self.w_o(attention_score)
    
    
    
    
    
    
    
class LayerNormalization(nn.Module):
    def __init__(self,epsillon: float = 1e-8):
        super().__init__()
        self.epsillon = epsillon
        self.aplha = nn.Parameter(torch.ones(size=1))
        self.beta  = nn.Parameter(torch.ones(size=1))
        # TODO : Dig in more about this one .... 
    def forward(self,x:torch.Tensor):
        mean = x.mean(dim = -1,keepdim=True) # x ---> (batch,row,col)
        std  = x.std(dim = -1,keepdim=True) 
        normalized = self.aplha * (x-mean)/(std + self.epsillon) + self.beta
        return normalized


