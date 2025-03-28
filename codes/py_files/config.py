from pathlib import Path
from typing import TypedDict
# ```type: ignore```  is added to avoid seeing the warnings on my code editor ... nothing related to the code's working or anything.  
class Config(TypedDict): # just for the sake of type checking and a better autocomplete for the developer :P 
    
    huggingface_dataset_download_input : str # type: ignore
    dataset_folder_name : Path  # type: ignore
    target_language : str  # type: ignore
    source_language : str  # type: ignore
    max_seq_len : int
    d_model : int 
    tokenizer_file : str
def get_config()->Config:
    return {
        "huggingface_dataset_download_input" : "nateraw/english-to-hinglish",
        "dataset_folder_name" : Path("hi_en-to-en"),
        "target_language" : "en",
        "source_language" : "hi_en",
        "max_seq_len" : 512, # subject to change depending upon the max value of the sequence encountered during training. 
        "d_model":512, 
        "tokenizer_file": "tokenizer-{lang}-.json",        
    }
    
 