from pathlib import Path
from typing import Callable
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import Dataset,DataLoader,random_split
from config import Config, get_config
from datasets import load_dataset
def get_all_sentences(ds, lang):
    for item in ds:
        yield item[lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(model = BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace() #type:ignore 
        
        trainer = WordLevelTrainer(special_tokens =["[UNK]", "[PAD]", "[SOS]", "[EOS]"],#type:ignore 
                                   min_frequency=2)#type:ignore 
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config:Config):
    dataset_path = Path(config['dataset_folder_name'])
    if not dataset_path.exists():
        dataset_ = load_dataset(config['huggingface_dataset_download_input'])
        dataset_.save_to_disk(dataset_path) # type: ignore
    
    