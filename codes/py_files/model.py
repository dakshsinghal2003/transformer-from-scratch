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
        self.positional_embeddings = positional_embeddings.unsqueeze(0)
        self.register_buffer(name='positional_embeddings',tensor=self.positional_embeddings)
        
        
    def forward(self,sequence:torch.Tensor):
        if sequence.ndim==2:
            sequence =sequence.unsqueeze(0)
        assert sequence.shape[2] == self.d_model,f"""The embedding dimensions of model and input dont match model's dimensions : {self.d_model} , input dimensions : {sequence.shape[0]}"""
        sequence =  sequence + (self.positional_embeddings[:,:sequence.shape[1],:]).requires_grad_(False)
        return self.dropout(sequence)

class FeedForwardNeuralNetwork(nn.Module):
    def __init__(self,input_dim:int = 512, output_dim = 2048, p_dropout:float = 0.1):
        super().__init__()
        self.linear_layer1 = nn.Linear(in_features=input_dim,out_features=output_dim)
        self.linear_layer2 = nn.Linear(in_features=output_dim,out_features=input_dim)
        self.dropout = nn.Dropout(p = p_dropout)
    def forward(self,x:torch.Tensor):
        return self.linear_layer2(self.dropout(self.linear_layer1(x)))
    
    
from torch import nn
import torch 
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
        q_q_head_wise = q_w.view(batch,sequence_length,self.h,self.d_key).transpose(2,1) 
        q_k_head_wise = q_w.view(batch,sequence_length,self.h,self.d_key).transpose(2,1) 
        q_v_head_wise = q_w.view(batch,sequence_length,self.h,self.d_key).transpose(2,1) 
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
    def forward(self,x:torch.Tensor):
        mean = x.mean(dim = -1,keepdim=True)
        std  = x.std(dim = -1,keepdim=True) 
        normalized = self.aplha * (x-mean)/(std + self.epsillon) + self.beta
        return normalized





class ResidualConnection(nn.Module):
    def __init__(self):
        super().__init__()
        self.normalize = LayerNormalization(epsillon=1e-8)
    def forward(self,x:torch.Tensor,sub_layer):
        # here the logic of sublayer is that the operations are performed on the input using some layer be if feed-forward or muti-head attention these 2 shall act as sublayers and then provide the output for it. Once we have the output we can continue with the addition of output and the input and then normalize them as the name says "add and norm" in the paper.  
        return self.normalize(x + sub_layer(x))

class EncoderBlock(nn.Module):
    def __init__(self,
                 d_model : int ,
                 h : int , 
                 dropout : float,
                 feed_forward_output : int , 
                ):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model,h=h,dropout=dropout)
        self.feed_forward_block = FeedForwardNeuralNetwork(input_dim=d_model,
                                                           output_dim = feed_forward_output,
                                                           p_dropout=dropout)
        # this is done to actually provide more flexibility for using skip connections properly. Since these skip connections are to be used 2 times here that too in between of serveral other connections. So we're essentially using it for skipping out on extra components and just focus on the residual connection.  
        
        self.skip_connections  = nn.ModuleList([ResidualConnection() for _ in range(2)])
        
    def forward(self,x:torch.Tensor,mask:torch.Tensor):
        # first we pass the input to the multi-head attention layer. 
        x1 = self.skip_connections[0](x,lambda x: self.self_attention(x,x,x,mask))
        x2 = self.skip_connections[1](x1,lambda x1:self.feed_forward_block(x1))
        return x2        
    
    
    


class Encoder(nn.Module):
    def __init__(self,
                 num_stacks: int = 6,
                 d_model : int = 512 , 
                 h : int = 8 , 
                 dropout : float = 0.1 , 
                 feed_forward_output : int = 2048 
                ):
        super().__init__()
        self.normalize = LayerNormalization()
        
        self.encoder_list = nn.ModuleList([EncoderBlock(d_model=d_model,
                                                        h = h,
                                                        dropout=dropout,
                                                        feed_forward_output=feed_forward_output)
                                           for _ in range(num_stacks)])
    
    def forward(self,x,mask):
        for layers in self.encoder_list:
            x = layers(x,mask)
        return self.normalize(x) # TODO : Check out in which part the normalization at this step is prescribed. Add that into the notes.  
    
class DecoderBlock(nn.Module):
    def __init__(self,
                 d_model : int ,
                 h : int , 
                 dropout : float,
                 feed_forward_output : int , 
                ) -> None:
        super().__init__(self )
        self.self_attention = MultiHeadAttention(
                                                 d_model=d_model,
                                                 h = h,
                                                 dropout=dropout
                                                )
        
        self.cross_attention = MultiHeadAttention(
                                                 d_model=d_model,
                                                 h = h,
                                                 dropout=dropout
                                                 )
        
        self.feed_forward_layers = FeedForwardNeuralNetwork(
                                                            input_dim=d_model,
                                                            output_dim=feed_forward_output,
                                                            p_dropout=dropout
                                                           )
        
        self.residual_connection = nn.ModuleList([ResidualConnection() for _ in range(3)])
        
    def forward(self,x:torch.Tensor,
                encoder_outputs : torch.Tensor,
                source_mask:torch.Tensor,
                target_mask:torch.Tensor,
                ):
        # TODO : understand the working of this source_mask and target_mask . this is something that I didn't find in the paper nor the tutorial explained quite clear
        x1 = self.residual_connection[0](x , lambda x: self.self_attention(x,x,x,target_mask))
        # TODO : Check this one out : What I've understood from the varius sources is that the encoder output is used as the query and key values and the encoded values produced by the decoder's self_attention are used as the value vector. In the function implementaiton of multihead attention, we've used the sequecne of query key value as input, but in the tutorial they've use the  same sequnce but while calling it here in the second part where we needed the values from the decoder as well in the form of value vector. In the tutorial they passed it as x1,encoder_output , encoder_output which is kind of wrong. 
        
        
        x2 = self.residual_connection[1](x1, lambda x1:self.cross_attention(encoder_outputs,encoder_outputs,x1,source_mask))
        x3 = self.residual_connection[2](x2,lambda x2 : self.feed_forward_layers(x2))
        return x3
    
    
class Decoder(nn.Module):
    def __init__(self,
                 num_stack:int = 6,
                 d_model : int = 512,
                 h : int = 8, 
                 dropout : float = 0.1,
                 feed_forward_output : int = 2048 ,
                 ) -> None:
        super().__init__()
        self.normalize = LayerNormalization()
        
        self.decoder_list = nn.ModuleList([DecoderBlock(
                                                        d_model=d_model,
                                                        h = h,
                                                        dropout=dropout,
                                                        feed_forward_output=feed_forward_output
                                                      ) for _ in range(num_stack)])
    
    def forward(self,x,encoder_output,source_mask,target_mask):
        for layer in self.decoder_list:
            x = layer(x,encoder_output,source_mask,target_mask)
        return self.normalize(x) # TODO : Check out in which part the normalization at this step is prescribed. Add that into the notes.         

class ProjectionLayer(nn.Module):
    def __init__(self, vocab_size : int , d_model : int = 512) -> None:
        super().__init__()
        self.linear_layer = nn.Linear(in_features=d_model,out_features=vocab_size)
        
    
    def forward(self,x:torch.Tensor):
        return torch.log_softmax(self.linear_layer(x),dim = -1)
    

class Transformer(nn.Module):
    def __init__(self,
                 encoder:Encoder,
                 decoder:Decoder,
                 source_embedding: InputEmbeddings,
                 target_embedding: InputEmbeddings,
                 source_positional_encoding:PositionalEncoding,
                 target_positional_encoding:PositionalEncoding,
                 projection_layer:ProjectionLayer
                ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.source_embedding = source_embedding
        self.target_embedding = target_embedding
        self.source_positional_encoding = source_positional_encoding
        self.target_positional_encoding = target_positional_encoding
        self.projection_layer = projection_layer
    def encode(self,
               source,
               source_mask
              ):
        input_embeddings = self.source_embedding(source)
        input_embeddings_with_positional_encodings = self.source_positional_encoding(input_embeddings)
        encodings = self.encoder(input_embeddings_with_positional_encodings,
                                 source_mask)
        return encodings
    
    
    def decode(self,encoder_output,source_mask,target,target_mask):
        target_embeddings = self.target_embedding(target)
        target_embeddings_with_positional_encodings = self.target_positional_encoding(target_embeddings)
        decodings = self.decoder(target_embeddings_with_positional_encodings,
                                 encoder_output,
                                 source_mask,
                                 target_mask)
        return decodings
    
    def project(self,x):
        return self.projection_layer(x)
    

def build_transformer(input_vocab_size:int,
                      target_vocab_size:int,
                      max_input_seq_lenght : int ,
                      max_target_seq_lenght : int ,
                      d_model : int  = 512,
                      dropout : float = 0.1, 
                      num_stacks : int = 6, 
                      num_attention_heads : int = 8, 
                      feed_forward_num_out : int = 2048)->Transformer:
    encoder= Encoder(
                      num_stacks=num_stacks,
                      d_model=d_model,
                      h = num_attention_heads,
                      dropout=dropout,
                      feed_forward_output=feed_forward_num_out)
    decoder= Decoder(num_stack=num_stacks,
                     d_model=d_model,
                     h=num_attention_heads,
                     dropout=dropout,
                     feed_forward_output=feed_forward_num_out)
    source_embedding=  InputEmbeddings(
                                        d_model=d_model,
                                        vocab_size=input_vocab_size
                                      )
    target_embedding=  InputEmbeddings(
                                        d_model=d_model,
                                        vocab_size=target_vocab_size
                                      )
    source_positional_encoding = PositionalEncoding(
                                                     d_model=d_model,
                                                     sequence_length=max_input_seq_lenght,
                                                     p_drop=dropout
                                                   )
    target_positional_encoding= PositionalEncoding(
                                                    d_model=d_model,
                                                    sequence_length=max_target_seq_lenght
                                                  )
    projection_layer= ProjectionLayer(  
                                        vocab_size=target_vocab_size,
                                        d_model=d_model
                                     )
    
    transformer_object = Transformer(
                                      encoder=encoder,
                                      decoder=decoder,
                                      source_embedding=source_embedding,
                                      target_embedding=target_embedding,
                                      source_positional_encoding=source_positional_encoding,
                                      target_positional_encoding=target_positional_encoding,
                                      projection_layer=projection_layer)
    
    
    # for p in transformer_object.parameters():
    #   print(p)
    return transformer_object 