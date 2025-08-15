from pathlib import Path
from typing import TypedDict
# ```type: ignore```  is added to avoid seeing the warnings on my code editor ... nothing related to the code's working or anything.  
class Config(TypedDict): # just for the sake of type checking and a better autocomplete for the developer :P 
    
    # huggingface_dataset_download_input : str # type: ignore
    dataset_folder_name : Path  # type: ignore
    target_language : str  # type: ignore
    source_language : str  # type: ignore
    max_seq_len : int
    d_model : int 
    tokenizer_file : str
    train_test : float
    train_validate : float
    batch_size : int
    dropout: float
    num_atttention_heads:int
    num_stacks : int
    max_token_length : int # can use this to limit the token size in BPE Trainer
    feed_forward_num_out : int
    
def get_config()->Config:
    return {
        # "huggingface_dataset_download_input" : "nateraw/english-to-hinglish",
        "dataset_folder_name" : Path("./hinglish-to-english-dataset"),
        "target_language" : "en_query",
        "source_language" : "cs_query",
        "max_seq_len" : 336, # subject to change depending upon the max value of the sequence encountered during training. 
        "d_model":512, 
        "tokenizer_file": "tokenizer-{lang}-.json",  
        "train_test": 0.9,
        "train_validate":0.1, 
        "batch_size":10, # since my dataset is 1,78,701 . I can safely ignore the last entry and then continue with my batch size of 10 i.e I'll update my parameters 17870 times in a single epoch . Note that I need to skip the rest of the values for calculating the max sequence length of the input and target tokens. 
        "dropout":0.1,
        "num_atttention_heads":8,
        "num_stacks":6,
        "max_token_length":6,
        "feed_forward_num_out":2048,
    }
    
 