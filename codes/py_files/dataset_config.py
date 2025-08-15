from torch.utils.data import dataset,DataLoader,Dataset
import torch
from typing import Any
def hidding_future(size:int):
    mask=  torch.triu(input=torch.ones(size = (1,size,size),dtype=torch.int64),diagonal=1).int()
    return mask==0
    
    



class BiLingualDataset(Dataset):
    def __init__(self,
                 dataset_,
                 source_tokenizer,
                 target_tokenizer,
                 source_language,
                 target_language,
                 max_seq_length
                ) :
        super().__init__()
        self.dataset_ = dataset_
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.source_language = source_language
        self.target_language = target_language
        self.max_seq_length = max_seq_length
        
        self.sos_token = torch.tensor([source_tokenizer.token_to_id("[SOS]")], dtype=torch.int64)
        
        self.eos_token = torch.tensor([source_tokenizer.token_to_id("[EOS]")], dtype=torch.int64)
        
        self.pad_token = torch.tensor([source_tokenizer.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.dataset_)
    
    def __getitem__(self, index) -> Any:
        source_target_pair = self.dataset_[index]
        souce = source_target_pair[self.source_language] # for my test case it's `hi_ng`
        target = source_target_pair[self.target_language] # for my test case it's `en`
        encoder_input_tokens = self.source_tokenizer.encode(souce).ids
        decoder_input_tokens = self.target_tokenizer.encode(target).ids
        
        
        
        encoder_pad_tokens = self.max_seq_length - len(encoder_input_tokens) - 2 
        # - 2 for adding the [SOS] and [EOS] token. 
        decoder_pad_tokens = self.max_seq_length - len(decoder_input_tokens) - 1 
        # - 1 only here and not -2 cause in decoder we only add  [SOS] token. 
        if encoder_pad_tokens< 0 :
            raise ValueError("Input sequece too long :( ")
        if decoder_pad_tokens < 0 :
            raise ValueError("Target sequece too long :( ")
        
        encoder_input = torch.cat(
                                    [
                                        self.sos_token,
                                        torch.tensor(data = encoder_input_tokens,dtype=torch.int64),
                                        self.eos_token,
                                        torch.tensor([self.pad_token]* encoder_pad_tokens,dtype=torch.int64)    
                                    ]
                                 )
        decoder_input = torch.cat(
                                    [
                                        self.sos_token,
                                        torch.tensor(data = decoder_input_tokens,dtype=torch.int64),
                                        torch.tensor([self.pad_token]* decoder_pad_tokens,dtype=torch.int64)
                                    ]
                                 )
        # for decoder there's no [EOS] token. TODO : Find out why but don't fuck around for long. 
        
        label = torch.cat(
                            [
                                torch.tensor(data = decoder_input_tokens, dtype=torch.int64),
                                self.eos_token,
                                torch.tensor([self.pad_token]* decoder_pad_tokens,dtype=torch.int64)
                            ]
                         ) 
        # it's basically trying to pass on the message as to how we're expecting the 
        # output here . TODO : figure out why there's not [SOS] token in labels and only [EOS] token. 
        
        assert encoder_input.size(0) == self.max_seq_length , f"The final expected lenght for encoded tokens was {self.max_seq_length} but got {encoder_input.size(0)} instead."
        
        assert decoder_input.size(0) == self.max_seq_length , f"The final expected lenght for decoder was {self.max_seq_length} but got {decoder_input.size(0)} instead."
        
        assert label.size(0) == self.max_seq_length , f"The final expected lenght for the label was {self.max_seq_length} but got {label.size(0)} instead."
        
        # if all good , return all the shit you've calculated so far 
        
        return {
                    "encoder_input":encoder_input,
                    "decoder_input":decoder_input,
                    "label":label,
                    "encoder_mask": (encoder_input!=self.pad_token).unsqueeze(0).unsqueeze(0).int(),
                    # these 2 extra unsqueeze are to add an extra 1 dimension to make the final shape as (1,1,seq_length) 
                    "decoder" : (decoder_input!=self.pad_token).unsqueeze(0).unsqueeze(0).int() &  hidding_future(decoder_input.size(0)),
                    # here we're using 2 sets of masks , the first one is to save the decoder from reading into the [PAD] tokens and the second is to stop the decoder from peeping into the future tokens. 
                    # generally speaking , this hiding_future(decoder_input.size(0)) will always return self.sequence_lenght because we ensure that the length of the decoder input is always self.sequence_length. 
                    "label" : label,
                    "source_text":souce,
                    "target_text":target,
               }
        